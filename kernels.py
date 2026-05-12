"""
File: Klernels.py
Authors: Clement Plessis - Martin Ecarnot
Date: 11 Oct. 2024
Description: This script is part of the IHS SPIKE project 
and is designed to segment and get the morphology of
wheat kernels after the
hyperspectral data acquistion.

"""

# ===================================
#       IMPORT
import os, cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from segment_anything import SamAutomaticMaskGenerator, modeling
from skimage.measure import regionprops
import gzip

# ===================================
#       CLASS
class Kernels():
    def __init__(
        self,
        image_rgb: np.ndarray,
        sam_model: modeling.sam.Sam,
        crop_x_left: int, crop_x_right: int
    ):
        mask_generator = SamAutomaticMaskGenerator(
            model=sam_model, box_nms_thresh=0.25
        )
        self.image_rgb = image_rgb[:,crop_x_left:crop_x_right]
        self.masks = mask_generator.generate(self.image_rgb)

    def filter_masks(self, area_min: int=4000, area_max: int=15000) -> None:
        """
        Filter masks by their surface.

        Arguments:
            masks (list): List of masks from 'mask_generator.generate'.

        Returns:
            list : List of masks from 'mask_generator.generate'.
        """
        new_masks = []
        for m in self.masks:
            if m['area'] > area_min and m['area'] < area_max:
                new_masks.append(m)

        new_masks = sorted(new_masks, key=(lambda x: x['bbox']), reverse=True)
        self.masks = new_masks

    def save_masks(self, output_path: str, sample: str, date: str, hour: str) -> None:
        img = self.image_rgb
        if len(self.masks) == 0:
            return
        plt.figure(figsize=(22,16))
        plt.imshow(img)
        ax = plt.gca()
        ax.set_autoscale_on(False)
        img = np.ones((self.masks[0]['segmentation'].shape[0], self.masks[0]['segmentation'].shape[1], 4))
        img[:,:,3] = 0
        n = 0
        for ann in self.masks:
            n+=1
            m = ann['segmentation']
            color_mask = np.concatenate([np.random.random(3), [0.45]])
            img[m] = color_mask
            bbox = ann['bbox']
            coord=( (bbox[0]*2+bbox[2]) / 2, (bbox[1]*2 + bbox[3]) / 2 )
            plt.text(x=int(coord[0]), y=int(coord[1]), s=str(n), fontsize=10, color= (1,1,1))
        ax.imshow(img)
        plt.axis('off')
        plt.savefig(f"{output_path}/{date}_{hour}_{sample}_masks.jpg",bbox_inches='tight')
        plt.close()

    def add_regionprops(self):
        props = list()
        for i in self.masks:
            p = regionprops(i["segmentation"]*255)[0]
            props.append(p)
        self.rprops = props

    def save_rpops(self, output_path: str, sample: str, date: str, hour: str) -> None:
        attrok = (
            'area', 'area_bbox', 'area_convex', 'eccentricity',
            'equivalent_diameter_area', 'euler_number', 'extent',
            'feret_diameter_max', 'area_filled', 'label',
            'axis_major_length', 'axis_minor_length', 'orientation',
            'perimeter', 'perimeter_crofton', 'solidity'
        )
        res_list = list()
        for i, props in enumerate(self.rprops):
            
            mask = self.masks[i]["segmentation"]
            img = self.image_rgb
            b, g, r = cv2.split(img)
            
            i_dict = dict(
                date = date,
                hour = hour,
                sample = sample,
                kernel = i+1,
                blue_min = np.min(b[mask]),
                blue_mean = np.mean(b[mask]),
                blue_max = np.max(b[mask]),
                green_min = np.min(g[mask]),
                green_mean = np.mean(g[mask]),
                green_max = np.max(g[mask]),
                red_min = np.min(r[mask]),
                red_mean = np.mean(r[mask]),
                red_max = np.max(r[mask])
            )
            for key in attrok:
                i_dict[key] = props[key]
            res_list.append(i_dict)

        df = pd.DataFrame(res_list)
        df.to_csv(f"{output_path}/{date}_{hour}_{sample}_props.csv",index=False)
    
    def save_kernels(self, output_path: str, sample: str, date: str, hour: str,
        size: int=320,resize: bool=True) -> None:
        if not os.path.isdir(f"{output_path}/kernels"):
            os.mkdir(f"{output_path}/kernels")

        for i, m in enumerate(self.masks):
            # Float to int
            img = (self.image_rgb * 255).astype(np.uint8)

            # Mask bool to int
            mask_int = m['segmentation'].astype(np.uint8)

            # Étendre le masque pour qu'il ait 3 canaux, car l'image est en RGB (3 canaux)
            mask_rgb = np.repeat(mask_int[:, :, np.newaxis], 3, axis=2)

            # Appliquer le masque sur l'image RGB pour extraire la zone correspondante
            masked_image = cv2.bitwise_and(img, img, mask=mask_int)

            # Définition de la bounding box (x_min, y_min, largeur, hauteur)
            x_min = m['bbox'][0]   # Coordonnée x du coin supérieur gauche
            y_min = m['bbox'][1]   # Coordonnée y du coin supérieur gauche
            largeur = m['bbox'][2] # Largeur de la boîte
            hauteur = m['bbox'][3] # Hauteur de la boîte

            add_width = int(round((size-largeur)/2,0))
            add_height = int(round((size-hauteur)/2,0))

            y1 = y_min-add_height
            y2 = y_min+hauteur+add_height
            x1=x_min-add_width
            x2=x_min+largeur+add_width

            if y1 < 0:
                y2 = y2+(0-y1)
                y1 = 0

            im_height = masked_image.shape[0]
            if y2 > im_height:
                y1 = y1-(y2-im_height)

            if x1 < 0 :
                x2 = x2+(0-x1)
                x1 = 0

            im_length = masked_image.shape[1]
            if x2 > im_length:
                x1 = x1-(x2-im_length)

            # Resize for model
            if resize:
                res = cv2.resize(masked_image[y1:y2,x1:x2], dsize=(640, 640), interpolation=cv2.INTER_CUBIC)
                plt.imsave(f'{output_path}/kernels/{date}_{hour}_{sample}_k{i+1}.jpg', res)
            else:
                plt.imsave(f'{output_path}/kernels/{date}_{hour}_{sample}_k{i+1}.jpg', masked_image[y1:y2,x1:x2])
            plt.close()

    # def save_Kernelspectra(self, ihsr, output_path: str, sample: str, date: str, hour: str):
    #     print(f"ihsr: {ihsr}, output_path: {output_path}, sample: {sample}, date: {date}, hour: {hour}")

    def save_Kernelspectra(self, ihsr, ref, output_path: str, sample: str, date: str, hour: str):

        sp = np.empty((0,ihsr.shape[2]+3)).astype(np.int16)
        spm = np.empty((0,ihsr.shape[2]+3))
        dep = np.reshape(ihsr, (ihsr.shape[0] * ihsr.shape[1], ihsr.shape[2]))  # unfolded image

        for i,m in enumerate(self.masks):
            # Coordinates of pixels of the mask
            xy_coords = np.column_stack(np.where(m["segmentation"] > 0))

            # Coord of grains pixels in unfolded image
            id = np.ravel_multi_index(np.transpose(xy_coords),(ihsr.shape[0], ihsr.shape[1]))

            # Fill sp with coodinates and spectra
            sp1 = np.array([dep[j, :] for j in id]).astype(np.int16)
            spcoord = np.concatenate((np.full((len(id), 1), i + 1),xy_coords, sp1),axis=1).astype(np.int16)
            sp = np.concatenate((sp, spcoord)) # Mandatory of (())

            # Convert to reflectance then average
            spcoord=spcoord.astype(np.float64)
            for j in range(ref.shape[0]):  # Parcours de chaque ligne de spref
                iok = spcoord[:, 1] == j  # Trouver les lignes correspondant à la ligne j dans sp
                if np.any(iok):  # Si des correspondances existent
                     spcoord[iok, 3:] = spcoord[iok, 3:] / ref[j, :][np.newaxis, :]  # Normalisation des spectres

            spm = np.vstack((spm, spcoord.mean(axis=0)))

        with gzip.GzipFile(f"{output_path}/{date}_{hour}_{sample}_sp_allpx.gz", "wb") as f:
            np.save(f, sp)

        with gzip.GzipFile(f"{output_path}/{date}_{hour}_{sample}_sp.gz", "wb") as f:
            np.save(f, spm)

        with gzip.GzipFile(f"{output_path}/{date}_{hour}_{sample}_ref.gz", "wb") as f:
            np.save(f, ref)

    def RebuildFromSpectra(self):  # noqa: N802
        # Définir les dimensions des matrices
        imred = np.ones((sp[:, 2].max(), sp[:, 1].max()))  # Matrice initialisée à 1
        imgr = np.zeros((sp[:, 2].max(), sp[:, 1].max()))  # Matrice initialisée à 0
        imbl = np.zeros((sp[:, 2].max(), sp[:, 1].max()))  # Matrice initialisée à 0

        # Remplir les matrices
        for i in range(sp.shape[0]):
            # Remplir les canaux R, G, et B
            imred[int(sp[i, 2]) - 1, int(sp[i, 1]) - 1] = sp[i, 82]  # Attention à l'indice Python (0-based)
            imgr[int(sp[i, 2]) - 1, int(sp[i, 1]) - 1] = sp[i, 54]
            imbl[int(sp[i, 2]) - 1, int(sp[i, 1]) - 1] = sp[i, 14]

        # Créer une image RGB en combinant les matrices
        imrgb = np.stack((imred, imgr, imbl), axis=2)

        # Afficher l'image
        plt.imshow(imrgb)  # Conversion en entier si nécessaire
        plt.axis('off')
        plt.show()


# ===================================
#       YOLO TWO-STAGE CLASS
CLASS_NAMES = {0: "grain", 1: "broken"}
CLASS_COLORS = {0: (0, 200, 0), 1: (0, 0, 220)}   # grain=vert, broken=rouge


class KernelsYOLO:
    """
    Two-stage YOLO pipeline that mirrors the Kernels (SAM) interface.

    Stage 1 — YOLO-det on the full (cropped) image → bounding boxes.
    Stage 2 — YOLO-seg on each grain crop          → mask + class (grain/broken).

    Each entry in self.masks is a dict compatible with the SAM format:
        {
            'segmentation': np.ndarray bool (H, W),
            'bbox':         [x_min, y_min, width, height],   # pixel coords
            'area':         int,
            'class':        int,   # 0=grain, 1=broken
            'confidence':   float,
        }
    """

    def __init__(
        self,
        image_rgb: np.ndarray,
        det_model,
        seg_model,
        crop_x_left: int,
        crop_x_right: int,
        det_conf: float = 0.3,
        seg_conf: float = 0.25,
        pad: float = 0.25,
        imgsz_det: int = 1280,
        imgsz_seg: int = 640,
        device: str = "0",
    ):
        # Crop to ROI and convert to uint8 for YOLO
        img_float = image_rgb[:, crop_x_left:crop_x_right]
        if img_float.dtype != np.uint8:
            img_uint8 = (np.clip(img_float, 0, 1) * 255).astype(np.uint8)
        else:
            img_uint8 = img_float
        self.image_rgb = img_float
        self._img_uint8 = img_uint8

        H, W = img_uint8.shape[:2]

        # Stage 1 — detect all grains
        det_res = det_model.predict(
            img_uint8, conf=det_conf, imgsz=imgsz_det,
            device=device, verbose=False
        )[0]

        self.masks = []
        if det_res.boxes is None or len(det_res.boxes) == 0:
            return

        for box in det_res.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            # Pad bbox
            bw, bh = x2 - x1, y2 - y1
            px, py = bw * pad, bh * pad
            cx1 = max(0, int(x1 - px))
            cy1 = max(0, int(y1 - py))
            cx2 = min(W, int(x2 + px))
            cy2 = min(H, int(y2 + py))
            cw, ch = cx2 - cx1, cy2 - cy1
            if cw < 8 or ch < 8:
                continue

            crop = img_uint8[cy1:cy2, cx1:cx2]

            # Stage 2 — segment the crop
            seg_res = seg_model.predict(
                crop, conf=seg_conf, imgsz=imgsz_seg,
                device=device, verbose=False
            )[0]

            if seg_res.masks is None or len(seg_res.masks) == 0:
                continue

            # Keep highest-confidence instance (central grain)
            best_idx = int(seg_res.boxes.conf.argmax())
            cls = int(seg_res.boxes[best_idx].cls.item())
            conf = float(seg_res.boxes[best_idx].conf.item())

            mask_crop = seg_res.masks.data[best_idx].cpu().numpy()
            mask_crop = cv2.resize(mask_crop, (cw, ch),
                                   interpolation=cv2.INTER_NEAREST)
            binary = mask_crop > 0.5

            # Reproject into full ROI coordinates
            full_mask = np.zeros((H, W), dtype=bool)
            full_mask[cy1:cy2, cx1:cx2] = binary

            self.masks.append({
                "segmentation": full_mask,
                "bbox": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
                "area": int(binary.sum()),
                "class": cls,
                "confidence": conf,
            })

    # ------------------------------------------------------------------
    #   Public interface — same as Kernels (SAM)
    # ------------------------------------------------------------------

    def filter_masks(self, area_min: int = 4000, area_max: int = 15000) -> None:
        self.masks = [
            m for m in self.masks
            if area_min < m["area"] < area_max
        ]
        self.masks = sorted(self.masks, key=lambda x: x["bbox"], reverse=True)

    def save_masks(self, output_path: str, sample: str, date: str, hour: str) -> None:
        if not self.masks:
            return
        img_disp = self._img_uint8 if self._img_uint8.dtype == np.uint8 \
            else (np.clip(self.image_rgb, 0, 1) * 255).astype(np.uint8)
        H, W = img_disp.shape[:2]

        overlay = img_disp.copy().astype(np.float32)
        for m in self.masks:
            color = CLASS_COLORS.get(m["class"], (128, 128, 128))
            seg = m["segmentation"]
            for c, val in enumerate(color):
                overlay[:, :, c][seg] = val

        blended = cv2.addWeighted(img_disp.astype(np.float32), 0.55,
                                  overlay, 0.45, 0).astype(np.uint8)

        fig, ax = plt.subplots(figsize=(22, 16))
        ax.imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
        for n, m in enumerate(self.masks, 1):
            bbox = m["bbox"]
            cx = bbox[0] + bbox[2] / 2
            cy = bbox[1] + bbox[3] / 2
            ax.text(cx, cy, str(n), fontsize=10, color="white")
        ax.axis("off")
        fig.savefig(
            f"{output_path}/{date}_{hour}_{sample}_masks.jpg",
            bbox_inches="tight"
        )
        plt.close(fig)

    def add_regionprops(self) -> None:
        self.rprops = [
            regionprops(m["segmentation"].astype(np.uint8) * 255)[0]
            for m in self.masks
        ]

    def save_rpops(self, output_path: str, sample: str, date: str, hour: str) -> None:
        attrok = (
            "area", "area_bbox", "area_convex", "eccentricity",
            "equivalent_diameter_area", "euler_number", "extent",
            "feret_diameter_max", "area_filled", "label",
            "axis_major_length", "axis_minor_length", "orientation",
            "perimeter", "perimeter_crofton", "solidity",
        )
        img = self._img_uint8
        b, g, r = cv2.split(img)

        res_list = []
        for i, (m, props) in enumerate(zip(self.masks, self.rprops)):
            mask = m["segmentation"]
            i_dict = dict(
                date=date, hour=hour, sample=sample, kernel=i + 1,
                kernel_class=CLASS_NAMES.get(m["class"], "unknown"),
                confidence=round(m["confidence"], 3),
                blue_min=np.min(b[mask]),   blue_mean=np.mean(b[mask]),
                blue_max=np.max(b[mask]),
                green_min=np.min(g[mask]),  green_mean=np.mean(g[mask]),
                green_max=np.max(g[mask]),
                red_min=np.min(r[mask]),    red_mean=np.mean(r[mask]),
                red_max=np.max(r[mask]),
            )
            for key in attrok:
                i_dict[key] = props[key]
            res_list.append(i_dict)

        pd.DataFrame(res_list).to_csv(
            f"{output_path}/{date}_{hour}_{sample}_props.csv", index=False
        )

    def save_kernels(self, output_path: str, sample: str, date: str, hour: str,
                     size: int = 320, resize: bool = True) -> None:
        os.makedirs(f"{output_path}/kernels", exist_ok=True)
        img = self._img_uint8

        for i, m in enumerate(self.masks):
            mask_int = m["segmentation"].astype(np.uint8)
            masked = cv2.bitwise_and(img, img, mask=mask_int)

            x_min, y_min, w, h = m["bbox"]
            add_w = int(round((size - w) / 2, 0))
            add_h = int(round((size - h) / 2, 0))
            y1, y2 = y_min - add_h, y_min + h + add_h
            x1, x2 = x_min - add_w, x_min + w + add_w

            im_h, im_w = masked.shape[:2]
            if y1 < 0:
                y2 += -y1; y1 = 0
            if y2 > im_h:
                y1 -= y2 - im_h
            if x1 < 0:
                x2 += -x1; x1 = 0
            if x2 > im_w:
                x1 -= x2 - im_w

            crop = masked[max(0, y1):y2, max(0, x1):x2]
            if resize:
                crop = cv2.resize(crop, (640, 640), interpolation=cv2.INTER_CUBIC)
            plt.imsave(
                f"{output_path}/kernels/{date}_{hour}_{sample}_k{i+1}.jpg",
                cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            )

    def save_Kernelspectra(self, ihsr, ref, output_path: str,  # noqa: N802
                            sample: str, date: str, hour: str) -> None:
        sp = np.empty((0, ihsr.shape[2] + 3)).astype(np.int16)
        spm = np.empty((0, ihsr.shape[2] + 3))
        dep = ihsr.reshape(ihsr.shape[0] * ihsr.shape[1], ihsr.shape[2])

        for i, m in enumerate(self.masks):
            xy_coords = np.column_stack(np.where(m["segmentation"] > 0))
            id_ = np.ravel_multi_index(
                np.transpose(xy_coords), (ihsr.shape[0], ihsr.shape[1])
            )
            sp1 = np.array([dep[j, :] for j in id_]).astype(np.int16)
            spcoord = np.concatenate(
                (np.full((len(id_), 1), i + 1), xy_coords, sp1), axis=1
            ).astype(np.int16)
            sp = np.concatenate((sp, spcoord))

            spcoord = spcoord.astype(np.float64)
            for j in range(ref.shape[0]):
                iok = spcoord[:, 1] == j
                if np.any(iok):
                    spcoord[iok, 3:] = spcoord[iok, 3:] / ref[j, :][np.newaxis, :]
            spm = np.vstack((spm, spcoord.mean(axis=0)))

        with gzip.GzipFile(f"{output_path}/{date}_{hour}_{sample}_sp_allpx.gz", "wb") as f:
            np.save(f, sp)
        with gzip.GzipFile(f"{output_path}/{date}_{hour}_{sample}_sp.gz", "wb") as f:
            np.save(f, spm)
        with gzip.GzipFile(f"{output_path}/{date}_{hour}_{sample}_ref.gz", "wb") as f:
            np.save(f, ref)