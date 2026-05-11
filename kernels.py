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
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
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

    def RebuildFromSpectra(self):
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