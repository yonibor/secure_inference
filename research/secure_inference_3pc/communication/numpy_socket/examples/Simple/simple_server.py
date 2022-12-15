#!/usr/bin/python3

import logging
import time

from research.secure_inference_3pc.communication.numpy_socket.numpysocket.numpysocket import NumpySocket

logger = logging.getLogger('simple server')
logger.setLevel(logging.INFO)

with NumpySocket() as s:
    s.bind(('', 9999))
    s.listen()
    conn, addr = s.accept()
    with conn:
        logger.info(f"connected: {addr}")
        frame = conn.recv()
        print(time.time())
        logger.info("array received")
        logger.info(frame)

    logger.info(f"disconnected: {addr}")
