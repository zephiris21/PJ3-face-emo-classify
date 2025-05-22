from typing import Optional, Union

import torch
import numpy as np
from codetiming import Timer
from PIL import Image
from facetorch.analyzer.predictor.core import FacePredictor
from facetorch.datastruct import ImageData, Response
from facetorch.logger import LoggerJsonFile
from importlib.metadata import version
from hydra.utils import instantiate
from omegaconf import OmegaConf

logger = LoggerJsonFile().logger


class FaceAnalyzer(object):
    @Timer(
        "FaceAnalyzer.__init__", "{name}: {milliseconds:.2f} ms", logger=logger.debug
    )
    def __init__(self, cfg: OmegaConf):
        """FaceAnalyzer is the main class that reads images, runs face detection, tensor unification and facial feature prediction.
        It also draws bounding boxes and facial landmarks over the image.

        The following components are used:

        1. Reader - reads the image and returns an ImageData object containing the image tensor.
        2. Detector - wrapper around a neural network that detects faces.
        3. Unifier - processor that unifies sizes of all faces and normalizes them between 0 and 1.
        4. Predictor dict - dict of wrappers around neural networks trained to analyze facial features.
        5. Utilizer dict - dict of utilizer processors that can for example extract 3D face landmarks or draw boxes over the image.

        Args:
            cfg (OmegaConf): Config object with image reader, face detector, unifier and predictor configurations.

        Attributes:
            cfg (OmegaConf): Config object with image reader, face detector, unifier and predictor configurations.
            reader (BaseReader): Reader object that reads the image and returns an ImageData object containing the image tensor.
            detector (FaceDetector): FaceDetector object that wraps a neural network that detects faces.
            unifier (FaceUnifier): FaceUnifier object that unifies sizes of all faces and normalizes them between 0 and 1.
            predictors (Dict[str, FacePredictor]): Dict of FacePredictor objects that predict facial features. Key is the name of the predictor.
            utilizers (Dict[str, FaceUtilizer]): Dict of FaceUtilizer objects that can extract 3D face landmarks, draw boxes over the image, etc. Key is the name of the utilizer.
            logger (logging.Logger): Logger object that logs messages to the console or to a file.

        """
        self.cfg = cfg
        self.logger = instantiate(self.cfg.logger).logger

        self.logger.info("Initializing FaceAnalyzer")
        self.logger.debug("Config", extra=self.cfg.__dict__["_content"])

        self.logger.info("Initializing BaseReader")
        self.reader = instantiate(self.cfg.reader)

        self.logger.info("Initializing FaceDetector")
        self.detector = instantiate(self.cfg.detector)

        self.logger.info("Initializing FaceUnifier")
        if "unifier" in self.cfg:
            self.unifier = instantiate(self.cfg.unifier)
        else:
            self.unifier = None

        self.logger.info("Initializing FacePredictor objects")
        self.predictors = {}
        if "predictor" in self.cfg:
            for predictor_name in self.cfg.predictor:
                self.logger.info(f"Initializing FacePredictor {predictor_name}")
                self.predictors[predictor_name] = instantiate(
                    self.cfg.predictor[predictor_name]
                )

        self.utilizers = {}
        if "utilizer" in self.cfg:
            self.logger.info("Initializing BaseUtilizer objects")
            for utilizer_name in self.cfg.utilizer:
                self.logger.info(f"Initializing BaseUtilizer {utilizer_name}")
                self.utilizers[utilizer_name] = instantiate(
                    self.cfg.utilizer[utilizer_name]
                )

    @Timer("FaceAnalyzer.run", "{name}: {milliseconds:.2f} ms", logger=logger.debug)
    def run(
        self,
        image_source: Optional[
            Union[str, torch.Tensor, np.ndarray, bytes, Image.Image]
        ] = None,
        path_image: Optional[str] = None,
        batch_size: int = 8,
        fix_img_size: bool = False,
        return_img_data: bool = False,
        include_tensors: bool = False,
        path_output: Optional[str] = None,
        tensor: Optional[torch.Tensor] = None,
    ) -> Union[Response, ImageData]:
        """Reads image, detects faces, unifies the detected faces, predicts facial features
         and returns analyzed data.

        Args:
            image_source (Optional[Union[str, torch.Tensor, np.ndarray, bytes, Image.Image]]): Input to be analyzed. If None, path_image or tensor must be provided. Default: None.
            path_image (Optional[str]): Path to the image to be analyzed. If None, tensor must be provided. Default: None.
            batch_size (int): Batch size for making predictions on the faces. Default is 8.
            fix_img_size (bool): If True, resizes the image to the size specified in reader. Default is False.
            return_img_data (bool): If True, returns all image data including tensors, otherwise only returns the faces. Default is False.
            include_tensors (bool): If True, removes tensors from the returned data object. Default is False.
            path_output (Optional[str]): Path where to save the image with detected faces. If None, the image is not saved. Default: None.
            tensor (Optional[torch.Tensor]): Image tensor to be analyzed. If None, path_image must be provided. Default: None.

        Returns:
            Union[Response, ImageData]: If return_img_data is False, returns a Response object containing the faces and their facial features. If return_img_data is True, returns the entire ImageData object.

        """

        def _predict_batch(
            data: ImageData, predictor: FacePredictor, predictor_name: str
        ) -> ImageData:
            n_faces = len(data.faces)

            for face_indx_start in range(0, n_faces, batch_size):
                face_indx_end = min(face_indx_start + batch_size, n_faces)

                face_batch_tensor = torch.stack(
                    [face.tensor for face in data.faces[face_indx_start:face_indx_end]]
                )
                preds = predictor.run(face_batch_tensor)
                data.add_preds(preds, predictor_name, face_indx_start)

            return data

        self.logger.info("Running FaceAnalyzer")

        if path_image is None and tensor is None and image_source is None:
            raise ValueError("Either input, path_image or tensor must be provided.")

        if image_source is not None:
            self.logger.debug("Using image_source as input")
            reader_input = image_source
        elif path_image is not None:
            self.logger.debug(
                "Using path_image as input", extra={"path_image": path_image}
            )
            reader_input = path_image
        else:
            self.logger.debug("Using tensor as input")
            reader_input = tensor

        self.logger.info("Reading image", extra={"input": reader_input})
        data = self.reader.run(reader_input, fix_img_size=fix_img_size)

        path_output = None if path_output == "None" else path_output
        data.path_output = path_output

        try:
            data.version = version("facetorch")
        except Exception as e:
            self.logger.warning("Could not get version number", extra={"error": e})

        self.logger.info("Detecting faces")
        data = self.detector.run(data)
        n_faces = len(data.faces)
        self.logger.info(f"Number of faces: {n_faces}")

        if n_faces > 0 and self.unifier is not None:
            self.logger.info("Unifying faces")
            data = self.unifier.run(data)

            self.logger.info("Predicting facial features")
            for predictor_name, predictor in self.predictors.items():
                self.logger.info(f"Running FacePredictor: {predictor_name}")
                data = _predict_batch(data, predictor, predictor_name)

            self.logger.info("Utilizing facial features")
            for utilizer_name, utilizer in self.utilizers.items():
                self.logger.info(f"Running BaseUtilizer: {utilizer_name}")
                data = utilizer.run(data)
        else:
            if "save" in self.utilizers:
                self.utilizers["save"].run(data)

        if not include_tensors:
            self.logger.debug(
                "Removing tensors from response as include_tensors is False"
            )
            data.reset_tensors()

        response = Response(faces=data.faces, version=data.version)

        if return_img_data:
            self.logger.debug("Returning image data object", extra=data.__dict__)
            return data
        else:
            self.logger.debug("Returning response with faces", extra=response.__dict__)
            return response
