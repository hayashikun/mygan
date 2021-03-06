import logging
import os

logging.basicConfig(level=logging.INFO)

PackageRoot = os.path.dirname(os.path.dirname(__file__))
ProjectResourcesRoot = PackageRoot

TmpFilePath = os.path.join(ProjectResourcesRoot, "tmp/")

if not os.path.exists(TmpFilePath):
    logging.info("tmp directory: {}".format(TmpFilePath))
    os.makedirs(TmpFilePath)
