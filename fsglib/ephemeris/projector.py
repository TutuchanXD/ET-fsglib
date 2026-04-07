import numpy as np

from fsglib.common.coords import radec_to_unit_vector, unit_vector_to_radec

class RealOpticalProjector:
    """
    Real projector incorporating Zemax cubic polynomial field distortion
    and ideal mounting rotations.
    """
    def __init__(self, cfg: dict):
        layout = cfg["layout"]
        self.projection_model = layout.get("projection_model", "distorted_focal_plane")
        self.pixel_size_mm = float(layout.get("pixel_size_mm", 1.0))
        
        # Detector definitions map (id -> dict)
        self.detectors_info = {}
        for det in layout["detectors"]:
            det_id = int(det["detector_id"])
            
            # Need numerical rotation matrices, user gave python arrays
            R = np.array(det["mounting_matrix"], dtype=np.float64) if det["mounting_matrix"] else np.eye(3)
            
            self.detectors_info[det_id] = {
                "principal_point_pix": np.array(det["principal_point_pix"], dtype=np.float64),
                "resolution": np.array(det["resolution"], dtype=np.float64),
                "mounting_matrix": R,
                "fov_center_mm": np.array(det.get("fov_center_mm", [0.0, 0.0]), dtype=np.float64),
                "pixel_scale_arcsec_per_pix": (
                    float(det["pixel_scale_arcsec_per_pix"])
                    if det.get("pixel_scale_arcsec_per_pix") is not None
                    else None
                ),
                "visibility_margin_pix": float(
                    det.get("visibility_margin_pix", layout.get("visibility_margin_pix", 0.0))
                ),
            }
            
        polys = layout.get("distortion", {})
        self.a1 = float(polys.get("a1", 0.0))
        self.a3 = float(polys.get("a3", 0.0))
        self.axy2 = float(polys.get("axy2", 0.0))
        self.field_center_ra_deg: float | None = None
        self.field_center_dec_deg: float | None = None
        self.field_offset_x_pix: float = 0.0
        self.field_offset_y_pix: float = 0.0
        self._field_center_c_ib: np.ndarray | None = None
        self._field_center_c_bi: np.ndarray | None = None

    def set_field_center(
        self,
        ra_deg: float | None,
        dec_deg: float | None,
        field_offset_x_pix: float | None = None,
        field_offset_y_pix: float | None = None,
    ) -> None:
        self.field_center_ra_deg = ra_deg
        self.field_center_dec_deg = dec_deg
        self.field_offset_x_pix = 0.0 if field_offset_x_pix is None else float(field_offset_x_pix)
        self.field_offset_y_pix = 0.0 if field_offset_y_pix is None else float(field_offset_y_pix)
        self._field_center_c_ib = None
        self._field_center_c_bi = None

        if ra_deg is None or dec_deg is None:
            return

        ra_rad = np.radians(float(ra_deg))
        dec_rad = np.radians(float(dec_deg))
        east = np.array([-np.sin(ra_rad), np.cos(ra_rad), 0.0], dtype=np.float64)
        north = np.array(
            [
                -np.sin(dec_rad) * np.cos(ra_rad),
                -np.sin(dec_rad) * np.sin(ra_rad),
                np.cos(dec_rad),
            ],
            dtype=np.float64,
        )
        center = np.array(
            [
                np.cos(dec_rad) * np.cos(ra_rad),
                np.cos(dec_rad) * np.sin(ra_rad),
                np.sin(dec_rad),
            ],
            dtype=np.float64,
        )
        self._field_center_c_ib = np.vstack([east, north, center])
        self._field_center_c_bi = self._field_center_c_ib.T

    def _image_to_field_deg(self, x_mm: float, y_mm: float) -> tuple[float, float]:
        """Forward: Image Plane (mm) -> Field Angle (deg)"""
        x = float(x_mm)
        y = float(y_mm)

        u = self.a1 * x + self.a3 * x**3 + self.axy2 * x * y**2
        v = self.a1 * y + self.a3 * y**3 + self.axy2 * y * x**2
        return u, v

    def _jacobian(self, x_mm: float, y_mm: float) -> np.ndarray:
        """Jacobian of the forward model w.r.t [x, y]."""
        x = float(x_mm)
        y = float(y_mm)

        du_dx = self.a1 + 3.0 * self.a3 * x**2 + self.axy2 * y**2
        du_dy = 2.0 * self.axy2 * x * y
        dv_dx = 2.0 * self.axy2 * x * y
        dv_dy = self.a1 + 3.0 * self.a3 * y**2 + self.axy2 * x**2

        return np.array([[du_dx, du_dy], [dv_dx, dv_dy]], dtype=float)

    def _field_deg_to_image(self, u_deg: float, v_deg: float, det_id: int, max_iter: int = 30, tol: float = 1e-10) -> tuple[float, float]:
        """Inverse: Field Angle (deg) -> Image Plane (mm) via Newton Iteration."""
        u_target = float(u_deg)
        v_target = float(v_deg)

        # Initial guess (center of the specific detector)
        x = self.detectors_info[det_id]["fov_center_mm"][0]
        y = self.detectors_info[det_id]["fov_center_mm"][1]

        for _ in range(max_iter):
            u_now, v_now = self._image_to_field_deg(x, y)
            residual = np.array([u_now - u_target, v_now - v_target], dtype=float)

            if np.linalg.norm(residual, ord=2) < tol:
                return x, y

            jacobian = self._jacobian(x, y)
            try:
                delta = np.linalg.solve(jacobian, residual)
            except np.linalg.LinAlgError:
                break # Failed to converge

            x -= delta[0]
            y -= delta[1]

            if np.linalg.norm(delta, ord=2) < tol:
                return x, y

        return None, None # Did not converge

    def _vector_to_field_deg(self, v_local: np.ndarray) -> tuple[float, float]:
        """Convert a local direction vector to (FOV_X, FOV_Y) in degrees (assuming standard convention)."""
        # Assume vector is [x, y, z], pointing towards +Z.
        # FOV angles typical approximation:
        v_norm = v_local / np.linalg.norm(v_local)
        
        # Field angles in radians
        # u = atan(x/z), v = atan(y/z)
        if v_norm[2] <= 0:
            return None, None
            
        u_rad = np.arctan2(v_norm[0], v_norm[2])
        v_rad = np.arctan2(v_norm[1], v_norm[2])
        return np.degrees(u_rad), np.degrees(v_rad)

    def pixel_to_los_body(self, detector_id: int, u_pix: float, v_pix: float) -> np.ndarray:
        """
        Step 1: Pixel -> local focal mm -> Field (u, v)
        Step 2: Field (u,v) -> Local 3D vector
        Step 3: Local 3D vector -> Body 3D vector  using mounting matrix
        """
        if detector_id not in self.detectors_info:
            return None
            
        info = self.detectors_info[detector_id]
        R = info["mounting_matrix"]

        if self.projection_model == "sky_patch_linearized":
            if self.field_center_ra_deg is None or self.field_center_dec_deg is None or self._field_center_c_ib is None:
                return None

            scale_arcsec = info["pixel_scale_arcsec_per_pix"]
            if scale_arcsec is None:
                return None

            x_deg = ((u_pix - self.field_offset_x_pix) - info["principal_point_pix"][0]) * scale_arcsec / 3600.0
            y_deg = ((v_pix - self.field_offset_y_pix) - info["principal_point_pix"][1]) * scale_arcsec / 3600.0
            dec0_rad = np.radians(self.field_center_dec_deg)
            ra_deg = self.field_center_ra_deg + x_deg / np.cos(dec0_rad)
            dec_deg = self.field_center_dec_deg + y_deg
            los_inertial = radec_to_unit_vector(ra_deg, dec_deg)
            return R @ (self._field_center_c_ib @ los_inertial)

        if self.projection_model == "ideal_pinhole":
            scale_arcsec = info["pixel_scale_arcsec_per_pix"]
            if scale_arcsec is None:
                return None

            u_deg = (u_pix - info["principal_point_pix"][0]) * scale_arcsec / 3600.0
            v_deg = (v_pix - info["principal_point_pix"][1]) * scale_arcsec / 3600.0

            u_rad = np.radians(u_deg)
            v_rad = np.radians(v_deg)
            tan_u = np.tan(u_rad)
            tan_v = np.tan(v_rad)
            z = 1.0 / np.sqrt(1.0 + tan_u**2 + tan_v**2)
            x = z * tan_u
            y = z * tan_v
            v_local = np.array([x, y, z], dtype=np.float64)
            return R @ v_local
        
        # 1. Pixel to mm (w.r.t optical axis 0,0)
        # Assuming pixel coordinates are 0-indexed, and principal point is relative to top-left.
        # Check signs carefully based on system conventions, here we assume standard +Right, +Down in image
        # mapping to physical +X, +Y in focal plane or similar. We will just use standard centered offset.
        dx_mm = (u_pix - info["principal_point_pix"][0]) * self.pixel_size_mm
        dy_mm = (v_pix - info["principal_point_pix"][1]) * self.pixel_size_mm
        
        # Add the detector's fov_center offsets to reference the global optical axis
        x_mm = dx_mm + info["fov_center_mm"][0]
        y_mm = dy_mm + info["fov_center_mm"][1]
        
        # 2. mm to Field Angles (u, v) deg
        u_deg, v_deg = self._image_to_field_deg(x_mm, y_mm)
        
        # 3. Field Angles to Local Vector
        u_rad = np.radians(u_deg)
        v_rad = np.radians(v_deg)
        
        # Reverse of atan
        # z = 1 / sqrt(1 + tan(u)^2 + tan(v)^2)
        tan_u = np.tan(u_rad)
        tan_v = np.tan(v_rad)
        z = 1.0 / np.sqrt(1.0 + tan_u**2 + tan_v**2)
        x = z * tan_u
        y = z * tan_v
        
        v_local = np.array([x, y, z], dtype=np.float64)
        
        # 4. Local Vector to Body Vector
        v_body = R @ v_local
        
        return v_body

    def los_body_to_pixel(self, detector_id: int, v_body: np.ndarray) -> tuple[float, float]:
        """
        Step 1: Body 3D -> Local 3D using inverse mounting
        Step 2: Local 3D -> Field (u,v)
        Step 3: Field (u,v) -> global mm (Newton Iteration)
        Step 4: global mm -> local mm -> Pixel
        """
        if detector_id not in self.detectors_info:
            return None, None
            
        info = self.detectors_info[detector_id]
        R = info["mounting_matrix"]
        R_inv = R.T # Assuming Orthogonal
        v_local = R_inv @ v_body

        if self.projection_model == "sky_patch_linearized":
            if self.field_center_ra_deg is None or self.field_center_dec_deg is None or self._field_center_c_bi is None:
                return None, None

            scale_arcsec = info["pixel_scale_arcsec_per_pix"]
            if scale_arcsec is None or scale_arcsec <= 0:
                return None, None

            los_inertial = self._field_center_c_bi @ v_local
            los_inertial /= np.linalg.norm(los_inertial)
            ra_deg, dec_deg = unit_vector_to_radec(los_inertial)

            ra0_rad = np.radians(self.field_center_ra_deg)
            dec0_rad = np.radians(self.field_center_dec_deg)
            ra_rad = np.radians(ra_deg)
            dec_rad = np.radians(dec_deg)
            dra_rad = np.arctan2(np.sin(ra_rad - ra0_rad), np.cos(ra_rad - ra0_rad))
            x_deg = np.degrees(dra_rad * np.cos(dec0_rad))
            y_deg = np.degrees(dec_rad - dec0_rad)

            u_pix = info["principal_point_pix"][0] + (x_deg * 3600.0 / scale_arcsec) + self.field_offset_x_pix
            v_pix = info["principal_point_pix"][1] + (y_deg * 3600.0 / scale_arcsec) + self.field_offset_y_pix
            return float(u_pix), float(v_pix)

        if self.projection_model == "ideal_pinhole":
            u_deg, v_deg = self._vector_to_field_deg(v_local)
            if u_deg is None:
                return None, None

            scale_arcsec = info["pixel_scale_arcsec_per_pix"]
            if scale_arcsec is None or scale_arcsec <= 0:
                return None, None

            u_pix = info["principal_point_pix"][0] + (u_deg * 3600.0 / scale_arcsec)
            v_pix = info["principal_point_pix"][1] + (v_deg * 3600.0 / scale_arcsec)
            return float(u_pix), float(v_pix)
            
        # 1. Body to Local
        # 2. Local to Field Angles
        u_deg, v_deg = self._vector_to_field_deg(v_local)
        if u_deg is None:
            return None, None
            
        # 3. Field Angles to global mm
        x_mm, y_mm = self._field_deg_to_image(u_deg, v_deg, detector_id)
        if x_mm is None:
            return None, None
            
        # 4. Global mm to local detector mm, then to Pixels
        dx_mm = x_mm - info["fov_center_mm"][0]
        dy_mm = y_mm - info["fov_center_mm"][1]
        
        u_pix = (dx_mm / self.pixel_size_mm) + info["principal_point_pix"][0]
        v_pix = (dy_mm / self.pixel_size_mm) + info["principal_point_pix"][1]
        
        return float(u_pix), float(v_pix)

    def project_to_detectors(self, los_inertial: np.ndarray, attitude_q: np.ndarray) -> tuple[dict, dict, list]:
        """
        Projects an inertial vector into the camera planes considering the full attitude matrix.
        Returns predicted_xy, predicted_valid, visible_det_ids.
        """
        # Convert quaternion to rotation matrix (Body to Inertial)
        q = attitude_q
        # Ensure q is scalar first: [w, x, y, z] is scipy standard (sometimes [x,y,z,w], assuming [w,x,y,z] here)
        from scipy.spatial.transform import Rotation
        # scipy uses [x, y, z, w], attitude solver outputs [w, x, y, z]
        rot = Rotation.from_quat([q[1], q[2], q[3], q[0]])
        C_IB = rot.as_matrix()
        
        v_body = C_IB @ los_inertial
        
        predicted_xy = {}
        predicted_valid = {}
        visible_det_ids = []
        
        for det_id, info in self.detectors_info.items():
            u, v = self.los_body_to_pixel(det_id, v_body)
            
            if u is None:
                predicted_valid[det_id] = False
                continue
                
            predicted_xy[det_id] = (u, v)
            
            # Check if within bounding box of the sensor resolution
            # adding small margin for edge effects if necessary, using raw bounds for now
            w, h = info["resolution"]
            margin = float(info.get("visibility_margin_pix", 0.0))
            if -margin <= u <= (w - 1 + margin) and -margin <= v <= (h - 1 + margin):
                predicted_valid[det_id] = True
                visible_det_ids.append(det_id)
            else:
                predicted_valid[det_id] = False
                
        return predicted_xy, predicted_valid, visible_det_ids
