import { cameraWithTensors } from "@tensorflow/tfjs-react-native";
import { Camera } from "expo-camera";
import { StatusBar } from "expo-status-bar";
import React, { useEffect, useState, useRef } from "react";
import {
  StyleSheet,
  Text,
  View,
  Platform,
  Dimensions,
  LogBox,
} from "react-native";
import * as cocoSsd from "@tensorflow-models/coco-ssd";
import { model } from "@tensorflow/tfjs";
import * as tf from "@tensorflow/tfjs";
import Canvas from "react-native-canvas";

const TensorCamera = cameraWithTensors(Camera);

const { width, height } = Dimensions.get("window");

export default function App() {
  const [model, setModel] = useState<cocoSsd.ObjectDetection>();
  let context = useRef<CanvasRenderingContext2D>();
  let canvas = useRef<Canvas>();

  let textureDims =
    Platform.OS === "ios"
      ? { height: 1920, width: 300 }
      : { height: 300, width: 300 };

  function handleCameraStream(images: any) {
    const loop = async () => {
      const nextImageTensor = images.next().value;
      if (!model || !nextImageTensor)
        throw new Error("No model or image tensor");
      model
        .detect(nextImageTensor)
        .then((predictions) => {
          drawRectangle(predictions, nextImageTensor);
        })
        .catch((error) => {
          console.log(error);
        });
      requestAnimationFrame(loop);
    };
    loop();
  }

  function drawRectangle(
    predictions: cocoSsd.DetectedObject[],
    nextImageTensor: any,
  ) {
    if (!context.current || !canvas.current) return;

    const scaleWidth = width / nextImageTensor.shape[1];
    const scaleHeight = height / nextImageTensor.shape[0];

    const flipHorizontal = Platform.OS == "ios" ? false : true;

    context.current.clearRect(0, 0, width, height);

    for (const prediction of predictions) {
      const [x, y, width, height] = prediction.bbox;

      const boundingBoxX = flipHorizontal
        ? canvas.current.width - x * scaleWidth - width * scaleWidth
        : x * scaleWidth;

      const boundingBoxY = y * scaleHeight;

      context.current.strokeRect(
        boundingBoxX,
        boundingBoxY,
        width * scaleWidth,
        height * scaleHeight,
      );

      context.current.strokeText(
        prediction.class,
        boundingBoxX - 5,
        boundingBoxY - 5,
      );
    }
  }

  async function handleCanvas(can: Canvas) {
    if (can) {
      can.width = width;
      can.height = height;
      const ctx: CanvasRenderingContext2D = can.getContext("2d");
      ctx.strokeStyle = "red";
      ctx.fillStyle = "red";
      ctx.lineWidth = 3;

      context.current = ctx;
      canvas.current = can;
    }
  }

  useEffect(() => {
    (async () => {
      const { status } = await Camera.requestCameraPermissionsAsync();
      await tf.ready();
      setModel(await cocoSsd.load());
    })();
  }, []);

  return (
    <View style={styles.container}>
      <TensorCamera
        style={styles.camera}
        type={Camera.Constants.Type.back}
        cameraTextureHeight={textureDims.height}
        cameraTextureWidth={textureDims.width}
        resizeHeight={200}
        resizeDepth={3}
        onReady={handleCameraStream}
        autorender={true}
        useCustomShadersToResize={false}
        resizeWidth={0}
      />
      <Canvas style={styles.canvas} ref={handleCanvas} />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#fff",
  },
  camera: { width: "100%", height: "100%" },
  canvas: {
    position: "absolute",
    zIndex: 1000000,
    width: "100%",
    height: "100",
  },
});
