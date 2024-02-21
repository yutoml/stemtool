import skimage.feature as skfeat
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as scnd
import scipy.optimize as spo
import scipy.interpolate as scinterp
import matplotlib_scalebar.scalebar as mpss
import stemtool as st
import warnings
from collections import UserList
import math
import os


class Peak():
    """
    ピークの座標や強度、近接ピーク間距離を保存する
    """

    def __init__(self, y: int | float, x: int | float, intensity: float | None = None) -> None:
        """
        Parameters
        ----------
        index0 : int
            画像データの0番目のindex
        index1 : int
            画像データの1番目のindex
        """
        self.x = x
        self.y = y
        self.intensity = intensity
        self.neighbers = []

    def get_distance(self, other):
        """Peak間の距離を返す。単位はpix

        Parameters
        ----------
        other : Peak
            自分自身と比較するpeak

        Returns
        -------
        float
            peak間の距離

        Raises
        ------
        TypeError
            otherがPeakではない場合にraiseされる
        """
        if not isinstance(other, self.__class__):
            raise TypeError("Value 'other' should be instance of class 'Peak'")

        return np.linalg.norm([self.x - other.x, self.y - other.y])

    def get_angle(self, other1, other2):
        """3つのpeakによって形成される角度を返す。
        頂点はselfとする
        すなわち, self -> other1とself -> other2のなす角である。

        Parameters
        ----------
        other1 : Peak
            計算に使用するpeak
        other2 : Peak
            計算に使用するpeak

        Returns
        -------
        float
            なす角. 単位はradianであり、0 ~ πの範囲である。

        Raises
        ------
        TypeError
            other1とother2がPeakではない場合にraiseされる
        """
        if not isinstance(other1, self.__class__) and not isinstance(other2, self.__class__):
            raise TypeError(
                "Value 'other1' and 'other2' should be instance of class 'Peak'")

        vector1 = [other1.x - self.x, other1.y - self.y]
        vector2 = [other2.x - self.x, other2.y - self.y]

        return np.arccos(np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2)))


class Peaks(UserList):
    """
    Peakのリスト
    近接原子間距離の計算を提供する
    """

    def __init__(self, initlist: list[Peak] | None = None):
        self.data: list[Peak] = []
        if initlist is not None:
            # XXX should this accept an arbitrary sequence?
            if type(initlist) == type(self.data):
                self.data[:] = initlist
            elif isinstance(initlist, UserList):
                self.data[:] = initlist.data[:]
            else:
                self.data = list(initlist)

    def __iter__(self):
        count_max = len(self.data)
        count = 0
        while True:
            if count >= count_max:
                break
            else:
                count += 1
                yield self.data[count - 1]

    def calc_neighbers(self, threshold: float = 1, division: bool = False):
        """thresholdより近いのピーク間を見つける

        Parameters
        ----------
        threshold : float, optional
            ピーク間距離の閾値。単位はpix, by default 1
        division : bool, optional
            ピークを分割して探索するピークを絞る。ピークが非常に多い場合は効果的, by default False

        """
        # 最初に近接原子間のパラメータを初期化する(再度計算していた場合に備えて)。
        for target in self.data:
            target.neighbers = []

        if division:
            # Peakをグループ分けして、計算量を減らす。
            # 各グループは1辺がthreshold以上の大きさの四角形を画像に敷き詰めたときにどの四角形に含まれるか、という形で分割される
            # 隣接するグループでのみピーク間距離を計算し、計算量を減らす
            x_list = [peak.x for peak in self.data]
            x_min, x_max = np.min(x_list), np.max(x_list)
            x_range = x_max - x_min
            x_division = math.floor(x_range / threshold)
            x_length = float(x_range / x_division)

            y_list = [peak.y for peak in self.data]
            y_min, y_max = np.min(y_list), np.max(y_list)
            y_range = y_max - y_min
            y_division = math.floor(y_range / threshold)
            y_length = float(y_range / y_division)

            groups = [[[] for j in range(x_division)]
                      for i in range(y_division)]

            for peak in self.data:
                y_index = round((peak.y - y_min) //
                                y_length) if peak.y != y_max else y_division - 1
                x_index = round((peak.x - x_min) //
                                x_length) if peak.x != x_max else x_division - 1
                groups[y_index][x_index].append(peak)
        else:
            y_division = 1
            x_division = 1
            groups = [[self.data]]
        processed = []

        def find_neighber_group_indices(y, x):
            """隣接するグループのインデックスを返す"""
            neighber_group_indices = []
            for i in [-1, 0, 1]:
                if y + i < 0 or y + i > y_division - 1:
                    continue
                for j in [-1, 0, 1]:
                    if x + j < 0 or x + j > x_division - 1:
                        continue
                    neighber_group_indices.append([y + i, x + j])

            return neighber_group_indices

        neighbers_count = []
        # グループ分けが終わったので、隣接ピークを見つける
        for y, row in enumerate(groups):
            for x, group in enumerate(row):
                neighber_group_indices = find_neighber_group_indices(y, x)

                while (group):
                    target = group.pop(0)
                    for neighber_group_index in neighber_group_indices:
                        for other in groups[neighber_group_index[0]][neighber_group_index[1]]:
                            if id(target) == id(other):  # targetとotherが同じ場合はスキップ
                                continue

                            if target.get_distance(other) < threshold:
                                target.neighbers.append(other)
                                other.neighbers.append(target)
                    processed.append(target)
                    neighbers_count.append(len(target.neighbers))
        self.data = processed
        print(f"Average_neighbers_num : {np.average(neighbers_count)}")
        return neighbers_count


class atom_fit(object):
    """
    Locate atom columns in atomic resolution STEM images
    and then subsequently use gaussian peak fitting to
    refine the column location with sub-pixel precision.

    Parameters
    ----------
    image:       ndarray
                 The image from which the peak positions
                 will be ascertained.
    calib:       float
                 Size of an individual pixel
    calib_units: str
                 Unit of calibration

    References
    ----------
    1]_, Mukherjee, D., Miao, L., Stone, G. and Alem, N.,
         mpfit: a robust method for fitting atomic resolution images
         with multiple Gaussian peaks. Adv Struct Chem Imag 6, 1 (2020).

    Examples
    --------
    Run as:

    >>> atoms = st.afit.atom_fit(stem_image, calibration, 'nm')

    Then to check the image you just loaded, with the optional parameter
    `12` determining how many pixels of gaussian blur need to applied to
    calculate and separate a background. If in doubt, don't use it.

    >>> atoms.show_image(12)

    It is then optional to define a refernce region for the image.
    If such a region is defined, atom positions will only be ascertained
    grom the reference region. If you don't run this step, the entire image
    will be analyzed

    >>> atoms.define_reference((17, 7), (26, 7), (26, 24), (17, 24))

    Then, visualize the peaks:

    >>> atoms.peaks_vis(dist=0.1, thresh=0.1)

    `dist` indicates the distance between the
    peaks in calibration units. Play around with the numbers till
    you get a satisfactory result. Then run the gaussian peak refinement as:

    >>> atoms.refine_peaks()

    You can visualize your fitted peaks as:

    >>> atoms.show_peaks(style= 'separate')

    """

    def __init__(self, image, calib, calib_units):
        self.image = st.util.image_normalizer(image)
        self.imcleaned = np.copy(self.image)
        self.calib = calib
        self.calib_units = calib_units
        self.imshape = np.asarray(image.shape)
        self.peaks_check = False
        self.refining_check = False
        self.reference_check = False

    def show_image(self, gaussval=0, imsize=(15, 15), colormap="inferno"):
        """
        Parameters
        ----------
        gaussval: int, optional
                  Extent of Gaussian blurring in pixels
                  to generate a background image for
                  subtraction. Default is 0
        imsize:   tuple, optional
                  Size in inches of the image with the
                  diffraction spots marked. Default is (15, 15)
        colormap: str, optional
                  Colormap of the image. Default is inferno
        """
        self.gaussval = gaussval
        if gaussval > 0:
            self.gblur = scnd.gaussian_filter(self.image, gaussval)
            self.imcleaned = st.util.image_normalizer(self.image - self.gblur)
        self.gauss_clean = gaussval
        plt.figure(figsize=imsize)
        plt.imshow(self.imcleaned, cmap=colormap)
        scalebar = mpss.ScaleBar(self.calib, self.calib_units)
        scalebar.location = "lower right"
        scalebar.box_alpha = 1
        scalebar.color = "k"
        plt.gca().add_artist(scalebar)
        plt.axis("off")

    def define_reference(self, A_pt, B_pt, C_pt, D_pt, imsize=(15, 15), tColor="k"):
        """
        Locate the reference image.

        Parameters
        ----------
        A_pt:   tuple
                Top left position of reference region in (x, y)
        B_pt:   tuple
                Top right position of reference region in (x, y)
        C_pt:   tuple
                Bottom right position of reference region in (x, y)
        D_pt:   tuple
                Bottom left position of reference region in (x, y)
        imsize: tuple, optional
                Size in inches of the image with the
                diffraction spots marked. Default is
                (10, 10)
        tColor: str, optional
                Color of the text on the image. Default is black

        Notes
        -----
        Locates a reference region bounded by the four points given in
        length units. Choose the points in a clockwise fashion.
        """
        A = np.asarray(A_pt) / self.calib
        B = np.asarray(B_pt) / self.calib
        C = np.asarray(C_pt) / self.calib
        D = np.asarray(D_pt) / self.calib

        yy, xx = np.mgrid[0: self.imshape[0], 0: self.imshape[1]]
        yy = np.ravel(yy)
        xx = np.ravel(xx)
        ptAA = np.asarray((xx, yy)).transpose() - A
        ptBB = np.asarray((xx, yy)).transpose() - B
        ptCC = np.asarray((xx, yy)).transpose() - C
        ptDD = np.asarray((xx, yy)).transpose() - D
        angAABB = np.arccos(
            np.sum(ptAA * ptBB, axis=1)
            / (
                ((np.sum(ptAA ** 2, axis=1)) ** 0.5)
                * ((np.sum(ptBB ** 2, axis=1)) ** 0.5)
            )
        )
        angBBCC = np.arccos(
            np.sum(ptBB * ptCC, axis=1)
            / (
                ((np.sum(ptBB ** 2, axis=1)) ** 0.5)
                * ((np.sum(ptCC ** 2, axis=1)) ** 0.5)
            )
        )
        angCCDD = np.arccos(
            np.sum(ptCC * ptDD, axis=1)
            / (
                ((np.sum(ptCC ** 2, axis=1)) ** 0.5)
                * ((np.sum(ptDD ** 2, axis=1)) ** 0.5)
            )
        )
        angDDAA = np.arccos(
            np.sum(ptDD * ptAA, axis=1)
            / (
                ((np.sum(ptDD ** 2, axis=1)) ** 0.5)
                * ((np.sum(ptAA ** 2, axis=1)) ** 0.5)
            )
        )
        angsum = ((angAABB + angBBCC + angCCDD + angDDAA) / (2 * np.pi)).reshape(
            self.imcleaned.shape
        )
        self.ref_reg = np.isclose(angsum, 1)
        self.ref_reg = np.flipud(self.ref_reg)

        pixel_list = np.arange(0, self.calib * self.imshape[0], self.calib)
        no_labels = 10
        step_x = int(self.imshape[0] / (no_labels - 1))
        x_positions = np.arange(0, self.imshape[0], step_x)
        x_labels = np.round(pixel_list[::step_x], 1)
        fsize = int(1.5 * np.mean(np.asarray(imsize)))

        print(
            "Choose your points in a clockwise fashion, or else you will get a wrong result"
        )

        plt.figure(figsize=imsize)
        plt.imshow(
            np.flipud(self.imcleaned + 0.33 * self.ref_reg),
            cmap="magma",
            origin="lower",
        )
        plt.annotate(
            "A=" + str(A_pt),
            A / self.imshape,
            textcoords="axes fraction",
            size=fsize,
            color=tColor,
        )
        plt.annotate(
            "B=" + str(B_pt),
            B / self.imshape,
            textcoords="axes fraction",
            size=fsize,
            color=tColor,
        )
        plt.annotate(
            "C=" + str(C_pt),
            C / self.imshape,
            textcoords="axes fraction",
            size=fsize,
            color=tColor,
        )
        plt.annotate(
            "D=" + str(D_pt),
            D / self.imshape,
            textcoords="axes fraction",
            size=fsize,
            color=tColor,
        )
        plt.scatter(A[0], A[1], c="r")
        plt.scatter(B[0], B[1], c="r")
        plt.scatter(C[0], C[1], c="r")
        plt.scatter(D[0], D[1], c="r")
        plt.xticks(x_positions, x_labels, fontsize=fsize)
        plt.yticks(x_positions, x_labels, fontsize=fsize)
        plt.xlabel("Distance along X-axis (" +
                   self.calib_units + ")", fontsize=fsize)
        plt.ylabel("Distance along Y-axis (" +
                   self.calib_units + ")", fontsize=fsize)
        self.reference_check = True

    def find_initial_peak(self, dist: float, thresh, gfilt=2, method: str = 'skfeat'):
        """のちのピークフィッティングで使う初期ピーク位置を極大値として探索する

        Parameters
        ----------
        dist : float
            peak間距離の閾値(pix単位ではなく、calib_unitsで指定した単位)
            この値より近いpeakは削除される。
        thresh : float
            peakの高さの閾値. 0 ~ 1.0の値で指定する
        gfilt : int, optional
            ガウシアンフィルターをかける範囲(pix単位), by default 2
        method : str, optional
            peak探索の方法, by default 'skfeat'
            MaximaFinderの場合は'fm2d'を指定する。

        Raises
        ------
        ValueError
            methodで存在しないメソッドを選択した場合
        """
        pixel_dist = dist / self.calib
        self.imfilt = scnd.gaussian_filter(self.imcleaned, gfilt)
        # 判定するべき範囲が指定されていない場合は、画像の全域が判定するべき範囲であるとする
        if not self.reference_check:
            self.ref_reg = np.ones_like(self.imcleaned, dtype=bool)
        # 判定するべき範囲以外は0にする
        data_ref = self.imfilt * self.ref_reg

        if method == "skfeat":
            self.positions = find_peaks_with_skfeat(
                data_ref, pixel_dist, thresh)

        elif method == "fm2d":
            self.positions = find_peaks_with_fm2d(data_ref, pixel_dist, thresh)

        else:
            raise ValueError(
                "You should choose one of the methods(skfeat or fm2d)")
        self.peaks = Peaks([Peak(position[0], position[1], intensity=self.image[int(position[0]), int(position[1])])
                            for position in self.positions])
        self.peaks_check = True

    def peaks_vis(self, imsize=(15, 15), spot_color="c"):
        """initial_peaksを表示する

        Parameters
        ----------
        imsize : tuple, optional
            画像のサイズ(インチ単位), by default (15, 15)
        spot_color : str, optional
            peak表示に使用する色, by default "c"
        """
        spot_size = int(0.5 * np.mean(np.asarray(imsize)))
        plt.figure(figsize=imsize)
        plt.imshow(self.imfilt, cmap="magma")
        plt.scatter([peak.x for peak in self.peaks], [peak.y for peak in self.peaks],
                    c=spot_color, s=spot_size)
        scalebar = mpss.ScaleBar(self.calib, self.calib_units)
        scalebar.location = "lower right"
        scalebar.box_alpha = 1
        scalebar.color = "k"
        plt.gca().add_artist(scalebar)
        plt.axis("off")

    def refine_peaks(self, do_test: bool = False, parallel: str | None = "multiprocessing", use_filtered: bool = False):
        if not self.peaks_check:
            raise RuntimeError(
                "Please locate the initial peaks first as peaks_vis()")
        if do_test:
            test = int(len(self.peaks) / 50)
            get_med_dist(self.peaks[0:test])
        md = get_med_dist(self.peaks)

        if use_filtered:
            img = self.imfilt
        else:
            img = self.imcleaned

        if do_test:
            # Run once on a smaller dataset to initialize JIT
            refined_positions = refine_atoms(
                img, self.peaks[0:test], md
            )

        # Run the JIT compiled faster code on the full dataset
        refined_positions = refine_atoms(img, self.peaks,
                                         md, parallel=parallel)
        self.refined_positions = refined_positions
        self.refined_peaks = Peaks([Peak(position[0], position[1], intensity=position[-1])
                                    for position in self.refined_positions])
        self.refining_check = True

    def mpfit(self, peak_runs=4, cut_point=1 / 3, tol_val=0.01, md_scale=1, max_workers=os.cpu_count() // 2, use_filtered: bool = False):
        """Multi-Gaussian Peak Refinement (mpfit)
        ガウシアンフィッティングを複数行うことで、原子の位置を決定する。

        Parameters
        ----------
        peak_runs : int, optional
            Number of multi-Gaussian steps to run, by default 4
        cut_point : _type_, optional
            Ratio of distance to the median inter-peak
            distance. Only Gaussian peaks below this are
            used for the final estimation, by default 1/3
        tol_val : float, optional
            The tolerance value to use for a gaussian estimation, by default 0.01
        md_scale : int, optional
            The scale factor for the size of Multi-Gaussian Peak(MGP) fitting area.
            md_scale = 1.2 means that the MGP fitting is processed with a square 
            whose length on one side is 1.2 times of the average interatomic distance., by default 1
        max_workers : int, optional
            The Number of processers for parallel calculation, by default half of the total CPU cores 

        Returns
        -------
        list[dict]
            The result of mpfit. Structure
            {
                "peak_runs" : [
                    [x, y, theta, sigma_x, sigma_y, amplitude],...
                ],
                "y_mpfit" : float, # peak_runsのピークの強さによる加重平均
                "x_mpfit" : float  # peak_runsのピークの強さによる加重平均
            }

        Notes
        -----
        This is the multiple Gaussian peak fitting technique
        where the initial atom positions are fitted with a
        single 2D Gaussian function. The calculated Gaussian is
        then subsequently subtracted and refined again. The final
        refined position is the sum of all the positions scaled
        with the amplitude

        References:
        -----------
        1]_, Mukherjee, D., Miao, L., Stone, G. and Alem, N.,
            mpfit: a robust method for fitting atomic resolution images
            with multiple Gaussian peaks. Adv Struct Chem Imag 6, 1 (2020).
        """
        if use_filtered:
            img = self.imfilt
        else:
            img = self.imcleaned

        results = mpfit(img, self.positions, peak_runs,
                        cut_point, tol_val, md_scale, max_workers)
        self.refined_positions = np.array(
            [np.array([result["y_mpfit"], result["x_mpfit"]]) for result in results])
        self.refined_peaks = Peaks([Peak(result["y_mpfit"], result["x_mpfit"], intensity=result["intensity"])
                                    for result in results])
        self.refining_check = True
        return results

    def show_peaks(self, imsize=(15, 15), style="together", use_filtered: bool = False):

        if not self.refining_check:
            raise RuntimeError(
                "Please refine the atom peaks first as refine_peaks()")

        if use_filtered:
            img = self.imfilt
        else:
            img = self.imcleaned

        togsize = tuple(np.asarray((2, 1)) * np.asarray(imsize))
        spot_size = int(0.5 * np.mean(np.asarray(imsize)))
        big_size = int(2 * spot_size)
        if style == "together":
            plt.figure(figsize=imsize)
            plt.imshow(img, cmap="magma")
            plt.scatter(
                [peak.x for peak in self.peaks],
                [peak.y for peak in self.peaks],
                c="c",
                s=big_size,
                label="Original Peaks",
            )
            plt.scatter(
                [peak.x for peak in self.refined_peaks],
                [peak.y for peak in self.refined_peaks],
                c="r",
                s=spot_size,
                label="Fitted Peaks",
            )
            plt.gca().legend(loc="upper left", markerscale=3, framealpha=1)
            scalebar = mpss.ScaleBar(self.calib, self.calib_units)
            scalebar.location = "lower right"
            scalebar.box_alpha = 1
            scalebar.color = "k"
            plt.gca().add_artist(scalebar)
            plt.axis("off")
        else:
            plt.figure(figsize=togsize)
            plt.subplot(1, 2, 1)
            plt.imshow(img, cmap="magma")
            plt.scatter(
                [peak.x for peak in self.peaks],
                [peak.y for peak in self.peaks],
                c="b",
                s=spot_size,
                label="Original Peaks",
            )
            plt.gca().legend(loc="upper left", markerscale=3, framealpha=1)
            scalebar = mpss.ScaleBar(self.calib, self.calib_units)
            scalebar.location = "lower right"
            scalebar.box_alpha = 1
            scalebar.color = "k"
            plt.gca().add_artist(scalebar)
            plt.axis("off")

            plt.subplot(1, 2, 2)
            plt.imshow(img, cmap="magma")
            plt.scatter(
                [peak.x for peak in self.refined_peaks],
                [peak.y for peak in self.refined_peaks],
                c="k",
                s=spot_size,
                label="Fitted Peaks",
            )
            plt.gca().legend(loc="upper left", markerscale=3, framealpha=1)
            scalebar = mpss.ScaleBar(self.calib, self.calib_units)
            scalebar.location = "lower right"
            scalebar.box_alpha = 1
            scalebar.color = "k"
            plt.gca().add_artist(scalebar)
            plt.axis("off")


def remove_close_vals(input_arr, limit):
    result = np.copy(input_arr)
    ii = 0
    newlen = len(result)
    while ii < newlen:
        dist = (
            np.sum(((result[:, 0:2] - result[ii, 0:2]) ** 2), axis=1)) ** 0.5
        distbool = dist > limit
        distbool[ii] = True
        result = np.copy(result[distbool, :])
        ii = ii + 1
        newlen = len(result)
    return result


def find_peaks_with_skfeat(data_image: np.ndarray, dist: float = 10, thresh: float = 0.1):
    """
    Find atom maxima pixels in images

    Parameters
    ----------
    data_image: ndarray
                Original atomic resolution image
    dist:       int
                Average distance between neighboring peaks
                Default is 10
    thresh:     float
                The cutoff intensity value below which a peak
                will not be detected
                Default is 0.1

    Returns
    -------
    peaks: ndarray
           List of peak positions as y, x

    Notes
    -----
    This is a wrapper around the skimage peak finding
    module which finds the peaks with a given threshold
    value and an inter-peak separation.

    :Authors:
    Debangshu Mukherjee <mukherjeed@ornl.gov>
    Yuto Ito <ito@stem.t.u-tokyo.ac.jp>
    """
    data_image = (data_image - np.amin(data_image)) / (
        np.amax(data_image) - np.amin(data_image)
    )
    thresh_arr = np.array(data_image > thresh, dtype=float)
    data_thresh = (data_image * thresh_arr)
    data_thresh = data_thresh / (1 - thresh)
    data_peaks = skfeat.peak_local_max(
        data_thresh, min_distance=int(dist / 3)
    )
    # 新しいscipyではmaskを返してくれなくなったので自作
    peak_mask = np.zeros_like(data_thresh, dtype=bool)
    peak_mask[tuple(data_peaks.T)] = True

    peak_labels = scnd.measurements.label(peak_mask)[0]
    merged_peaks = scnd.measurements.center_of_mass(
        peak_mask, peak_labels, range(1, np.max(peak_labels) + 1)
    )
    peaks = np.array(merged_peaks)
    peaks = remove_close_vals(peaks, dist)

    return peaks


def find_peaks_with_fm2d(data_image: np.ndarray, dist: float = 10, thresh: float = 0.1, mergin: int = 3):
    """
    Find atom maxima pixels in images with findmaxima2d by dwaithe

    Parameters
    ----------
    data_image: ndarray
                Original atomic resolution image
    dist:       int
                Average distance between neighboring peaks
                Default is 10
    thresh:     float
                The cutoff intensity value below which a peak
                will not be detected
                Default is 0.1
    thresh:     int
                the mergin between peak and border of image
                Default is 3

    Returns
    -------
    peaks: ndarray
           List of peak positions as y, x

    Notes
    -----
    This is a wrapper around the findmaxima2d peak finding
    module which finds the peaks with a given height of 
    peak and an inter-peak separation. 

    :Authors:
    Yuto Ito <ito@stem.t.u-tokyo.ac.jp>
    """
    data_image = (data_image - np.amin(data_image)) / (
        np.amax(data_image) - np.amin(data_image)
    )
    data_image_8bit = data_image * 255
    from findmaxima2d import find_maxima, find_local_maxima
    local_max = find_local_maxima(data_image_8bit)
    y, x, _ = find_maxima(data_image_8bit, local_max, thresh*255)
    # peakがボーダーにのっていないことを確認
    data_peaks = np.array([peak for peak in zip(y, x) if peak[0] > mergin and peak[1] >
                          mergin and peak[0] < data_image.shape[0]-1-mergin and peak[1] < data_image.shape[1]-1-mergin])

    peak_mask = np.zeros_like(data_image, dtype=bool)
    peak_mask[tuple(data_peaks.T)] = True

    peak_labels = scnd.measurements.label(peak_mask)[0]
    merged_peaks = scnd.measurements.center_of_mass(
        peak_mask, peak_labels, range(1, np.max(peak_labels) + 1)
    )
    peaks = np.array(merged_peaks)
    peaks = remove_close_vals(peaks, dist)

    return peaks


def mpfit_subroutine(main_image: np.ndarray, xx: np.ndarray, yy: np.ndarray, xstart: float, ystart: float, med_dist: float, peak_runs: int, tol_val: float, cut_point: float):

    sub_y = np.abs(yy - ystart) < med_dist  # 関心領域でTrueとなるy座標
    sub_x = np.abs(xx - xstart) < med_dist  # 関心領域でTrueとなるx座標
    sub = np.logical_and(sub_x, sub_y)  # 関心領域でTrueとなるx,y座標
    xvals = xx[sub]
    yvals = yy[sub]
    zvals = main_image[sub]  # 関心領域の画像データ
    zcalc = np.zeros_like(zvals)  # gaussianフィッティングで得られた強度
    cvals = np.zeros((peak_runs, 4), dtype=float)
    result = {"peak_runs": []}
    for ii in np.arange(peak_runs):
        zvals = zvals - zcalc
        zgaus = (zvals - np.amin(zvals)) / (np.amax(zvals) - np.amin(zvals))
        mask_radius = med_dist
        xy = (xvals, yvals)
        initial_guess = st.util.initialize_gauss2D(xvals, yvals, zgaus)
        lower_bound = (
            (initial_guess[0] - med_dist),
            (initial_guess[1] - med_dist),
            -180,
            0,
            0,
            ((-2.5) * initial_guess[5]),
        )
        upper_bound = (
            (initial_guess[0] + med_dist),
            (initial_guess[1] + med_dist),
            180,
            (2.5 * mask_radius),
            (2.5 * mask_radius),
            (2.5 * initial_guess[5]),
        )
        # poptはx0,y0,theta,sigmax,sigmay,amplitudeが入っているlist
        popt, _ = spo.curve_fit(
            st.util.gaussian_2D_function,
            xy,
            zgaus,
            initial_guess,
            bounds=(lower_bound, upper_bound),
            ftol=tol_val,
            xtol=tol_val,
        )
        result[f"peak_runs"].append(popt)
        cvals[ii, 1] = popt[0]  # x0
        cvals[ii, 0] = popt[1]  # y0
        cvals[ii, -1] = popt[-1] * \
            (np.amax(zvals) - np.amin(zvals))  # amplitude
        cvals[ii, 2] = (
            ((popt[0] - xstart) ** 2) + ((popt[1] - ystart) ** 2)  # 初期推定からのずれ
        ) ** 0.5
        zcalc = st.util.gaussian_2D_function(
            xy, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5]
        )
        zcalc = (zcalc * (np.amax(zvals) - np.amin(zvals))) + np.amin(zvals)
    required_cvals = cvals[:, 2] < (cut_point * med_dist)

    total = np.sum(cvals[required_cvals, 3])

    # それぞれのGausianfittingの結果の加重平均を最終結果とする
    result["y_mpfit"] = np.sum(
        cvals[required_cvals, 0] * cvals[required_cvals, 3]) / total
    result["x_mpfit"] = np.sum(
        cvals[required_cvals, 1] * cvals[required_cvals, 3]) / total
    result["intensity"] = total
    return result


def mpfit(
    main_image,
    initial_peaks,
    peak_runs=4,
    cut_point=1 / 3,
    tol_val=0.01,
    md_scale=1.0,
    max_workers=os.cpu_count() // 2
):
    """
    Multi-Gaussian Peak Refinement (mpfit)

    Parameters
    ----------
    main_image:     ndarray
                    Original atomic resolution image
    initial_peaks:  ndarray
                    Y and X position of maxima/minima
    peak_runs:      int
                    Number of multi-Gaussian steps to run
                    Default is 16
    cut_point:      float
                    Ratio of distance to the median inter-peak
                    distance. Only Gaussian peaks below this are
                    used for the final estimation
                    Default is 2/3
    tol_val:        float
                    The tolerance value to use for a gaussian estimation
                    Default is 0.01
    md_scale:       float
                    The scale factor for the size of Multi-Gaussian Peak(MGP) fitting area.
                    md_scale = 1.2 means that the MGP fitting is processed with a square 
                    whose length on one side is 1.2 times of the average interatomic distance.

    Returns
    -------
    mpfit_peaks: ndarray
                 List of refined peak positions as y, x

    Notes
    -----
    This is the multiple Gaussian peak fitting technique
    where the initial atom positions are fitted with a
    single 2D Gaussian function. The calculated Gaussian is
    then subsequently subtracted and refined again. The final
    refined position is the sum of all the positions scaled
    with the amplitude

    References:
    -----------
    1]_, Mukherjee, D., Miao, L., Stone, G. and Alem, N.,
         mpfit: a robust method for fitting atomic resolution images
         with multiple Gaussian peaks. Adv Struct Chem Imag 6, 1 (2020).
    """
    from concurrent.futures import ProcessPoolExecutor
    from tqdm import tqdm
    import itertools
    warnings.filterwarnings("ignore")

    med_dist = get_med_dist(initial_peaks) * md_scale
    yy, xx = np.mgrid[0: main_image.shape[0], 0: main_image.shape[1]]
    print(f"parallel calc ready, max_workers = {max_workers}")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # itertools.repeatをもちいてメモリを削減
        results = list(tqdm(executor.map(mpfit_subroutine, itertools.repeat(main_image), itertools.repeat(xx), itertools.repeat(yy), [pos[1] for pos in initial_peaks], [
                       pos[0] for pos in initial_peaks], itertools.repeat(med_dist), itertools.repeat(peak_runs), itertools.repeat(tol_val), itertools.repeat(cut_point)), desc="parallel calc started", total=len(initial_peaks)))
    return results


def get_med_dist(positions):
    """距離の中央値を計算 Calc the median of distances. distances are the distance of each peaks

    Args:
        positions (_type_): positions of peaks

    Returns:
        float: median of distance
    """
    warnings.filterwarnings("ignore")
    no_pos = len(positions)
    dist = np.empty(no_pos, dtype=float)
    ccd = np.empty(no_pos, dtype=float)
    for ii in np.arange(no_pos):
        ccd = np.sum(((positions[:, 0:2] - positions[ii, 0:2]) ** 2), axis=1)
        dist[ii] = (np.amin(ccd[ccd > 0])) ** 0.5
    med_dist = 0.5 * np.median(dist)
    return med_dist


def refine_atoms(image_data, positions, med_dist, parallel: str | None = "multiprocessing"):
    """原子位置をガウシアンフィッティングをかけてより正確に求める。

    Parameters
    ----------
    image_data : _type_
        画像データ
    positions : _type_
        原子位置
    ref_arr : _type_
        修正されて原子位置を保存するリスト. 空でよい
    med_dist : _type_
        平均原子間距離
    parallel : str | None, optional
        並列計算を行うか否か,並列計算を行う場合は'multiprocessing'を与え、逐次処理を行う場合はNoneを与える。デフォルトはNone(逐次処理)
    """
    ref_arr = np.zeros(shape=(len(positions), 7))
    if parallel == None:  # 並列計算なし
        for ii, (pos_x, pos_y) in enumerate(positions):
            refined_ii = st.util.fit_gaussian2D_mask(
                image_data, pos_x, pos_y, med_dist)
            ref_arr[ii, 0:2] = np.flip(refined_ii[0:2])
            ref_arr[ii, 2:6] = refined_ii[2:6]
            ref_arr[ii, -1] = refined_ii[-1]  # -1がされていたがいらない気がする

    elif parallel == "multiprocessing":  # multithprocessingを利用して高速化
        from concurrent.futures import ProcessPoolExecutor
        from tqdm import tqdm
        import itertools
        import os
        print(f"parallel calc ready, max_workers = {os.cpu_count() // 2}")
        with ProcessPoolExecutor(max_workers=os.cpu_count() // 2) as executor:
            # itertools.repeatをもちいてメモリを削減
            result = list(tqdm(executor.map(st.util.fit_gaussian2D_mask, itertools.repeat(image_data), [pos[1] for pos in positions], [
                          pos[0] for pos in positions], itertools.repeat(med_dist)), desc="parallel calc started", total=len(positions)))
        for ii, refined_ii in enumerate(result):
            ref_arr[ii, 0:2] = np.flip(refined_ii[0:2])
            ref_arr[ii, 2:6] = refined_ii[2:6]
            ref_arr[ii, -1] = refined_ii[-1]  # -1がされていたがいらない気がする
    else:
        raise ValueError(
            f"Option 'parallel' was given unsupported value. (parallel = {parallel})")
    return ref_arr


###############################################################################
# The codes below are not essential for the purpose of finding atom positions #
###############################################################################


def find_coords(image, fourier_center, fourier_y, fourier_x, y_axis, x_axis):
    """
    Convert the fourier positions to image axes.

    Parameters
    ----------
    image:  ndarray
            Original image
    four_c: ndarray
            Position of the central beam in
            the Fourier pattern
    four_y: ndarray
            Position of the y beam in
            the Fourier pattern
    four_x: ndarray
            Position of the x beam in
            the Fourier pattern

    Returns
    -------
    coords: ndarray
            Axes co-ordinates in the real image,
            as [y1 x1
                y2 x2]

    Notes
    -----
    Use the fourier coordinates to define the axes
    co-ordinates in real space, which will be used
    to assign each atom position to a axes position

    :Authors:
    Debangshu Mukherjee <mukherjeed@ornl.gov>
    """
    image_size = image.shape
    qx = (np.arange((-image_size[1] / 2),
          (image_size[1] / 2), 1)) / image_size[1]
    qy = (np.arange((-image_size[0] / 2),
          (image_size[0] / 2), 1)) / image_size[0]
    dx = np.mean(np.diff(qx))
    dy = np.mean(np.diff(qy))
    fourier_calib = (dy, dx)
    y_axis[y_axis == 0] = 1
    x_axis[x_axis == 0] = 1
    distance_y = fourier_y - fourier_center
    distance_y = np.divide(distance_y, y_axis[0:2])
    distance_x = fourier_x - fourier_center
    distance_x = np.divide(distance_x, x_axis[0:2])
    fourier_length_y = distance_y * fourier_calib
    fourier_length_x = distance_x * fourier_calib
    angle_y = np.degrees(
        np.arccos(fourier_length_y[0] / np.linalg.norm(fourier_length_y))
    )
    real_y_size = 1 / np.linalg.norm(fourier_length_y)
    angle_x = np.degrees(
        np.arccos(fourier_length_x[0] / np.linalg.norm(fourier_length_x))
    )
    real_x_size = 1 / np.linalg.norm(fourier_length_x)
    real_y = (
        (real_y_size * np.cos(np.deg2rad(angle_y - 90))),
        (real_y_size * np.sin(np.deg2rad(angle_y - 90))),
    )
    real_x = (
        (real_x_size * np.cos(np.deg2rad(angle_x - 90))),
        (real_x_size * np.sin(np.deg2rad(angle_x - 90))),
    )
    coords = np.asarray(((real_y[1], real_y[0]), (real_x[1], real_x[0])))
    if np.amax(np.abs(coords[0, :])) > np.amax(coords[0, :]):
        coords[0, :] = (-1) * coords[0, :]
    if np.amax(np.abs(coords[1, :])) > np.amax(coords[1, :]):
        coords[1, :] = (-1) * coords[1, :]
    return coords


def get_origin(image, peak_pos, coords):
    def origin_function(xyCenter, input_data=(peak_pos, coords)):
        peaks = input_data[0]
        coords = input_data[1]
        atom_coords = np.zeros((peaks.shape[0], 6))
        atom_coords[:, 0:2] = peaks[:, 0:2] - xyCenter[0:2]
        atom_coords[:, 2:4] = atom_coords[:, 0:2] @ np.linalg.inv(coords)
        atom_coords[:, 4:6] = np.round(atom_coords[:, 2:4])
        average_deviation = (
            ((np.mean(np.abs(atom_coords[:, 3] - atom_coords[:, 5]))) ** 2)
            + ((np.mean(np.abs(atom_coords[:, 2] - atom_coords[:, 4]))) ** 2)
        ) ** 0.5
        return average_deviation

    initial_x = image.shape[1] / 2
    initial_y = image.shape[0] / 2
    initial_guess = np.asarray((initial_y, initial_x))
    lower_bound = np.asarray(
        ((initial_y - initial_y / 2), (initial_x - initial_x / 2)))
    upper_bound = np.asarray(
        ((initial_y + initial_y / 2), (initial_x + initial_x / 2)))
    res = spo.minimize(
        fun=origin_function, x0=initial_guess, bounds=(
            lower_bound, upper_bound)
    )
    origin = res.x
    return origin


def get_coords(image, peak_pos, origin, current_coords):
    ang_1 = np.degrees(np.arctan2(current_coords[0, 1], current_coords[0, 0]))
    mag_1 = np.linalg.norm((current_coords[0, 1], current_coords[0, 0]))
    ang_2 = np.degrees(np.arctan2(current_coords[1, 1], current_coords[1, 0]))
    mag_2 = np.linalg.norm((current_coords[1, 1], current_coords[1, 0]))

    def coords_function(coord_vals, input_data=(peak_pos, origin, ang_1, ang_2)):
        mag_t = coord_vals[0]
        mag_b = coord_vals[1]
        peaks = input_data[0]
        rigin = input_data[1]
        ang_t = input_data[2]
        ang_b = input_data[3]
        xy_coords = np.asarray(
            (
                (mag_t * np.cos(np.deg2rad(ang_t)),
                 mag_t * np.sin(np.deg2rad(ang_t))),
                (mag_b * np.cos(np.deg2rad(ang_b)),
                 mag_b * np.sin(np.deg2rad(ang_b))),
            )
        )
        atom_coords = np.zeros((peaks.shape[0], 6))
        atom_coords[:, 0:2] = peaks[:, 0:2] - rigin[0:2]
        atom_coords[:, 2:4] = atom_coords[:, 0:2] @ np.linalg.inv(xy_coords)
        atom_coords[:, 4:6] = np.round(atom_coords[:, 2:4])
        average_deviation = (
            ((np.mean(np.abs(atom_coords[:, 3] - atom_coords[:, 5]))) ** 2)
            + ((np.mean(np.abs(atom_coords[:, 2] - atom_coords[:, 4]))) ** 2)
        ) ** 0.5
        return average_deviation

    initial_guess = np.asarray((mag_1, mag_2))
    lower_bound = initial_guess - (0.25 * initial_guess)
    upper_bound = initial_guess + (0.25 * initial_guess)
    res = spo.minimize(
        fun=coords_function, x0=initial_guess, bounds=(
            lower_bound, upper_bound)
    )
    mag = res.x
    new_coords = np.asarray(
        (
            (mag[0] * np.cos(np.deg2rad(ang_1)),
             mag[0] * np.sin(np.deg2rad(ang_1))),
            (mag[1] * np.cos(np.deg2rad(ang_2)),
             mag[1] * np.sin(np.deg2rad(ang_2))),
        )
    )
    return new_coords


def coords_of_atoms(peaks, coords, origin):
    """
    Convert atom positions to coordinates

    Parameters
    ----------
    peaks:  ndarray
            List of Gaussian fitted peaks
    coords: ndarray
            Co-ordinates of the axes

    Returns
    -------
    atom_coords: ndarray
                 Peak positions as the atom coordinates

    Notes
    -----
    One atom is chosen as the origin and the co-ordinates
    of all the atoms are calculated with respect to the origin

    :Authors:
    Debangshu Mukherjee <mukherjeed@ornl.gov>
    """
    atom_coords = np.zeros((peaks.shape[0], 8))
    atom_coords[:, 0:2] = peaks[:, 0:2] - origin[0:2]
    atom_coords[:, 2:4] = atom_coords[:, 0:2] @ np.linalg.inv(coords)
    atom_coords[:, 0:2] = peaks[:, 0:2]
    atom_coords[:, 4:6] = np.round(atom_coords[:, 2:4])
    atom_coords[:, 6:8] = atom_coords[:, 4:6] @ coords
    atom_coords[:, 6:8] = atom_coords[:, 6:8] + origin[0:2]
    return atom_coords


def fourier_mask(original_image, center, radius, threshold=0.2):
    image_fourier = np.fft.fftshift(np.fft.fft2(original_image))
    pos_x = center[0]
    pos_y = center[1]
    blurred_image = scnd.filters.gaussian_filter(np.abs(image_fourier), 3)
    fitted_diff = st.util.fit_gaussian2D_mask(
        blurred_image, pos_x, pos_y, radius)
    new_x = fitted_diff[0]
    new_y = fitted_diff[1]
    new_center = np.asarray((new_x, new_y))
    size_image = np.asarray(np.shape(image_fourier), dtype=int)
    yV, xV = np.mgrid[0: size_image[0], 0: size_image[1]]
    sub = ((((yV - new_y) ** 2) + ((xV - new_x) ** 2)) ** 0.5) < radius
    circle = np.copy(sub)
    circle = circle.astype(np.float64)
    filtered_circ = scnd.filters.gaussian_filter(circle, 1)
    masked_image = np.multiply(image_fourier, filtered_circ)
    SAED_image = np.fft.ifft2(masked_image)
    mag_SAED = np.abs(SAED_image)
    mag_SAED = (mag_SAED - np.amin(mag_SAED)) / \
        (np.amax(mag_SAED) - np.amin(mag_SAED))
    mag_SAED[mag_SAED < threshold] = 0
    mag_SAED[mag_SAED > threshold] = 1
    filtered_SAED = scnd.filters.gaussian_filter(mag_SAED, 3)
    filtered_SAED[filtered_SAED < threshold] = 0
    filtered_SAED[filtered_SAED > threshold] = 1
    fourier_selected_image = np.multiply(original_image, filtered_SAED)
    return fourier_selected_image, SAED_image, new_center, filtered_SAED


def find_diffraction_spots(image, circ_c, circ_y, circ_x):
    """
    Find the diffraction spots visually.

    Parameters
    ----------
    image:  ndarray
            Original image
    circ_c: ndarray
            Position of the central beam in
            the Fourier pattern
    circ_y: ndarray
            Position of the y beam in
            the Fourier pattern
    circ_x: ndarray
            Position of the x beam in
            the Fourier pattern


    Notes
    -----
    Put circles in red(central), y(blue) and x(green)
    on the diffraction pattern to approximately know
    the positions.

    :Authors:
    Debangshu Mukherjee <mukherjeed@ornl.gov>
    """
    image_ft = np.fft.fftshift(np.fft.fft2(image))
    log_abs_ft = scnd.filters.gaussian_filter(np.log10(np.abs(image_ft)), 3)
    f, ax = plt.subplots(figsize=(20, 20))
    circ_c_im = plt.Circle(circ_c, 15, color="red", alpha=0.33)
    circ_y_im = plt.Circle(circ_y, 15, color="blue", alpha=0.33)
    circ_x_im = plt.Circle(circ_x, 15, color="green", alpha=0.33)
    ax.imshow(log_abs_ft, cmap="gray")
    ax.add_artist(circ_c_im)
    ax.add_artist(circ_y_im)
    ax.add_artist(circ_x_im)
    plt.show()


def mpfit_voronoi(
    main_image,
    initial_peaks,
    peak_runs=16,
    cut_point=2 / 3,
    tol_val=0.01,
    blur_factor=0.25,
):
    """
    Multi-Gaussian Peak Refinement (mpfit)

    Parameters
    ----------
    main_image:     ndarray
                    Original atomic resolution image
    initial_peaks:  ndarray
                    Y and X position of maxima/minima
    peak_runs:      int
                    Number of multi-Gaussian steps to run
                    Default is 16
    cut_point:      float
                    Ratio of distance to the median inter-peak
                    distance. Only Gaussian peaks below this are
                    used for the final estimation
                    Default is 2/3
    tol_val:        float
                    The tolerance value to use for a gaussian estimation
                    Default is 0.01
    blur_factor:    float
                    Make the Voronoi regions slightly bigger.
                    Default is 25% bigger

    Returns
    -------
    mpfit_peaks: ndarray
                 List of refined peak positions as y, x

    Notes
    -----
    This is the multiple Gaussian peak fitting technique
    where the initial atom positions are fitted with a
    single 2D Gaussian function. The calculated Gaussian is
    then subsequently subtracted and refined again. The final
    refined position is the sum of all the positions scaled
    with the amplitude. The difference with the standard mpfit
    code is that the masking region is actually chosen as a
    Voronoi region from the nearest neighbors

    References:
    -----------
    1]_, Mukherjee, D., Miao, L., Stone, G. and Alem, N.,
         mpfit: a robust method for fitting atomic resolution images
         with multiple Gaussian peaks. Adv Struct Chem Imag 6, 1 (2020).

    :Authors:
    Debangshu Mukherjee <mukherjeed@ornl.gov>
    """
    warnings.filterwarnings("ignore")
    distm = np.zeros(len(initial_peaks))
    for ii in np.arange(len(initial_peaks)):
        ccd = np.sum(
            ((initial_peaks[:, 0:2] - initial_peaks[ii, 0:2]) ** 2), axis=1)
        distm[ii] = (np.amin(ccd[ccd > 0])) ** 0.5
    med_dist = np.median(distm)
    mpfit_peaks = np.zeros_like(initial_peaks, dtype=float)
    yy, xx = np.mgrid[0: main_image.shape[0], 0: main_image.shape[1]]
    cutoff = med_dist * 2.5
    for jj in np.arange(len(initial_peaks)):
        ypos, xpos = initial_peaks[jj, :]
        dist = (
            np.sum(
                ((initial_peaks[:, 0:2] - initial_peaks[jj, 0:2]) ** 2), axis=1)
        ) ** 0.5
        distn = dist < cutoff
        distn[dist < 0.1] = False
        neigh = initial_peaks[distn, 0:2]
        sub = (((yy - ypos) ** 2) + ((xx - xpos) ** 2)) < (cutoff ** 2)
        xvals = xx[sub]
        yvals = yy[sub]
        zvals = main_image[sub]
        maindist = ((xvals - xpos) ** 2) + ((yvals - ypos) ** 2)
        dist_mat = np.zeros((len(xvals), len(neigh)))
        for ii in np.arange(len(neigh)):
            dist_mat[:, ii] = ((xvals - neigh[ii, 1]) ** 2) + (
                (yvals - neigh[ii, 0]) ** 2
            )
        neigh_dist = np.amin(dist_mat, axis=1)
        voronoi = maindist < ((1 + blur_factor) * neigh_dist)
        xvor = xvals[voronoi]
        yvor = yvals[voronoi]
        zvor = zvals[voronoi]
        vor_dist = np.amax(
            (((xvor - xpos) ** 2) + ((yvor - ypos) ** 2)) ** 0.5)
        zcalc = np.zeros_like(zvor)
        xy = (xvor, yvor)
        cvals = np.zeros((peak_runs, 4), dtype=float)
        for ii in np.arange(peak_runs):
            zvor = zvor - zcalc
            zgaus = (zvor - np.amin(zvor)) / (np.amax(zvor) - np.amin(zvor))
            initial_guess = st.util.initialize_gauss2D(xvor, yvor, zgaus)
            lower_bound = (
                np.amin(xvor),
                np.amin(yvor),
                -180,
                0,
                0,
                ((-2.5) * initial_guess[5]),
            )
            upper_bound = (
                np.amax(xvor),
                np.amax(yvor),
                180,
                vor_dist,
                vor_dist,
                (2.5 * initial_guess[5]),
            )
            popt, _ = spo.curve_fit(
                st.util.gaussian_2D_function,
                xy,
                zgaus,
                initial_guess,
                bounds=(lower_bound, upper_bound),
                ftol=tol_val,
                xtol=tol_val,
            )
            cvals[ii, 1] = popt[0]
            cvals[ii, 0] = popt[1]
            cvals[ii, -1] = popt[-1] * (np.amax(zvor) - np.amin(zvor))
            cvals[ii, 2] = (((popt[0] - xpos) ** 2) +
                            ((popt[1] - ypos) ** 2)) ** 0.5
            zcalc = st.util.gaussian_2D_function(
                xy, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5]
            )
            zcalc = (zcalc * (np.amax(zvor) - np.amin(zvor))) + np.amin(zvor)
        required_cvals = cvals[:, 2] < (cut_point * vor_dist)
        total = np.sum(cvals[required_cvals, 3])
        y_mpfit = np.sum(cvals[required_cvals, 0] *
                         cvals[required_cvals, 3]) / total
        x_mpfit = np.sum(cvals[required_cvals, 1] *
                         cvals[required_cvals, 3]) / total
        mpfit_peaks[jj, 0:2] = np.asarray((y_mpfit, x_mpfit))
    return mpfit_peaks


def three_neighbors(peak_list, coords, delta=0.25):
    warnings.filterwarnings("ignore")
    no_atoms = peak_list.shape[0]
    atoms_neighbors = np.zeros((no_atoms, 8))
    atoms_distances = np.zeros((no_atoms, 4))
    for ii in range(no_atoms):
        atom_pos = peak_list[ii, 0:2]
        neigh_yy = atom_pos + coords[0, :]
        neigh_xx = atom_pos + coords[1, :]
        neigh_xy = atom_pos + coords[0, :] + coords[1, :]
        parnb_yy = ((peak_list[:, 0] - neigh_yy[0]) ** 2) + (
            (peak_list[:, 1] - neigh_yy[1]) ** 2
        )
        neigh_yy = (peak_list[parnb_yy == np.amin(parnb_yy), 0:2])[0]
        ndist_yy = np.linalg.norm(neigh_yy - atom_pos)
        parnb_xx = ((peak_list[:, 0] - neigh_xx[0]) ** 2) + (
            (peak_list[:, 1] - neigh_xx[1]) ** 2
        )
        neigh_xx = (peak_list[parnb_xx == np.amin(parnb_xx), 0:2])[0]
        ndist_xx = np.linalg.norm(neigh_xx - atom_pos)
        parnb_xy = ((peak_list[:, 0] - neigh_xy[0]) ** 2) + (
            (peak_list[:, 1] - neigh_xy[1]) ** 2
        )
        neigh_xy = (peak_list[parnb_xy == np.amin(parnb_xy), 0:2])[0]
        ndist_xy = np.linalg.norm(neigh_xy - atom_pos)
        atoms_neighbors[ii, :] = np.ravel(
            np.asarray((atom_pos, neigh_yy, neigh_xx, neigh_xy))
        )
        atoms_distances[ii, :] = np.ravel(
            np.asarray((0, ndist_yy, ndist_xx, ndist_xy)))
    yy_dist = np.linalg.norm(coords[0, :])
    yy_list = np.asarray(((yy_dist * (1 - delta)), (yy_dist * (1 + delta))))
    xx_dist = np.linalg.norm(coords[1, :])
    xx_list = np.asarray(((xx_dist * (1 - delta)), (xx_dist * (1 + delta))))
    xy_dist = np.linalg.norm(coords[0, :] + coords[1, :])
    xy_list = np.asarray(((xy_dist * (1 - delta)), (xy_dist * (1 + delta))))
    pp = atoms_distances[:, 1]
    pp[pp > yy_list[1]] = 0
    pp[pp < yy_list[0]] = 0
    pp[pp == 0] = np.nan
    atoms_distances[:, 1] = pp
    pp = atoms_distances[:, 2]
    pp[pp > xx_list[1]] = 0
    pp[pp < xx_list[0]] = 0
    pp[pp == 0] = np.nan
    atoms_distances[:, 2] = pp
    pp = atoms_distances[:, 3]
    pp[pp > xy_list[1]] = 0
    pp[pp < xy_list[0]] = 0
    pp[pp == 0] = np.nan
    atoms_distances[:, 3] = pp
    atoms_neighbors = atoms_neighbors[~np.isnan(atoms_distances).any(axis=1)]
    atoms_distances = atoms_distances[~np.isnan(atoms_distances).any(axis=1)]
    return atoms_neighbors, atoms_distances


def relative_strain(n_list, coords):
    warnings.filterwarnings("ignore")
    identity = np.asarray(((1, 0), (0, 1)))
    axis_pos = np.asarray(((0, 0), (1, 0), (0, 1), (1, 1)))
    no_atoms = (np.shape(n_list))[0]
    coords_inv = np.linalg.inv(coords)
    cell_center = np.zeros((no_atoms, 2))
    e_xx = np.zeros(no_atoms)
    e_xy = np.zeros(no_atoms)
    e_yy = np.zeros(no_atoms)
    e_th = np.zeros(no_atoms)
    for ii in range(no_atoms):
        cc = np.zeros((4, 2))
        cc[0, :] = n_list[ii, 0:2] - n_list[ii, 0:2]
        cc[1, :] = n_list[ii, 2:4] - n_list[ii, 0:2]
        cc[2, :] = n_list[ii, 4:6] - n_list[ii, 0:2]
        cc[3, :] = n_list[ii, 6:8] - n_list[ii, 0:2]
        l_cc, _, _, _ = np.linalg.lstsq(axis_pos, cc, rcond=None)
        t_cc = np.matmul(l_cc, coords_inv) - identity
        e_yy[ii] = t_cc[0, 0]
        e_xx[ii] = t_cc[1, 1]
        e_xy[ii] = 0.5 * (t_cc[0, 1] + t_cc[1, 0])
        e_th[ii] = 0.5 * (t_cc[0, 1] - t_cc[1, 0])
        cell_center[ii, 0] = 0.25 * (
            n_list[ii, 0] + n_list[ii, 2] + n_list[ii, 4] + n_list[ii, 6]
        )
        cell_center[ii, 1] = 0.25 * (
            n_list[ii, 1] + n_list[ii, 3] + n_list[ii, 5] + n_list[ii, 7]
        )
    return cell_center, e_yy, e_xx, e_xy, e_th


def strain_map(centers, e_yy, e_xx, e_xy, e_th, mask):
    yr, xr = np.mgrid[0: mask.shape[0], 0: mask.shape[1]]
    cartcoord = list(zip(centers[:, 1], centers[:, 0]))

    e_yy[np.abs(e_yy) > 3 * np.median(np.abs(e_yy))] = 0
    e_xx[np.abs(e_xx) > 3 * np.median(np.abs(e_xx))] = 0
    e_xy[np.abs(e_xy) > 3 * np.median(np.abs(e_xy))] = 0
    e_th[np.abs(e_th) > 3 * np.median(np.abs(e_th))] = 0

    f_yy = scinterp.LinearNDInterpolator(cartcoord, e_yy)
    f_xx = scinterp.LinearNDInterpolator(cartcoord, e_xx)
    f_xy = scinterp.LinearNDInterpolator(cartcoord, e_xy)
    f_th = scinterp.LinearNDInterpolator(cartcoord, e_th)

    map_yy = f_yy(xr, yr)
    map_yy[np.isnan(map_yy)] = 0
    map_yy = np.multiply(map_yy, mask)

    map_xx = f_xx(xr, yr)
    map_xx[np.isnan(map_xx)] = 0
    map_xx = np.multiply(map_xx, mask)

    map_xy = f_xy(xr, yr)
    map_xy[np.isnan(map_xy)] = 0
    map_xy = np.multiply(map_xy, mask)

    map_th = f_th(xr, yr)
    map_th[np.isnan(map_th)] = 0
    map_th = np.multiply(map_th, mask)

    return map_yy, map_xx, map_xy, map_th


def create_circmask(image, center, radius, g_val=3, flip=True):
    """
    Use a Gaussian blurred image to fit
    peaks.

    Parameters
    ----------
    image:  ndarray
            2D array representing the image
    center: tuple
            Approximate location as (x,y) of
            the peak we are trying to fit
    radius: float
            Masking radius
    g_val:  float
            Value in pixels of the Gaussian
            blur applied. Default is 3
    flip:   bool
            Switch to flip refined center position
            from (x,y) to (y,x). Default is True
            which returns the center as (y,x)


    Returns
    -------
    masked_image: ndarray
                  Masked Image centered at refined
                  peak position
    new_center:   ndarray
                  Refined atom center as (y,x) if
                  flip switch is on, else the center
                  is returned as (x,y)

    Notes
    -----
    For some noisy datasets, a peak may be visible with
    the human eye, but getting a sub-pixel estimation of
    the peak position is often challenging, especially
    for FFT for diffraction patterns. This code Gaussian
    blurs the image, and returns the refined peak position

    See also
    --------
    st.util.fit_gaussian2D_mask

    :Authors:
    Debangshu Mukherjee <mukherjeed@ornl.gov>

    """
    blurred_image = scnd.filters.gaussian_filter(np.abs(image), g_val)
    fitted_diff = st.util.fit_gaussian2D_mask(
        blurred_image, center[0], center[1], radius
    )
    size_image = np.asarray(np.shape(image), dtype=int)
    yV, xV = np.mgrid[0: size_image[0], 0: size_image[1]]
    masked_image = np.zeros_like(blurred_image)
    if flip:
        new_center = np.asarray(np.flip(fitted_diff[0:2]))
        sub = (
            (((yV - new_center[0]) ** 2) + ((xV - new_center[1]) ** 2)) ** 0.5
        ) < radius
    else:
        new_center = np.asarray(fitted_diff[0:2])
        sub = (
            (((yV - new_center[1]) ** 2) + ((xV - new_center[0]) ** 2)) ** 0.5
        ) < radius
    masked_image[sub] = image[sub]
    return masked_image, new_center
