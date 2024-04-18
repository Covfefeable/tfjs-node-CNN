import * as tf from "@tensorflow/tfjs-node";
import fs from "fs";
import path from "path";
import {
  imgToTensor,
  labelToStr,
  strToLabel,
} from "./const";

const config = {
  dataSetPath: "./images/new-train/",
  modelPath: "file://./model",
  imageWidth: 80,
  imageHeight: 34,
}

const train = async () => {
  // 导入所有验证码数据集
  const imagesDataset = fs.readdirSync(config.dataSetPath);
  // labels 用于存储验证码的标签
  const labels: number[][] = [];

  const imagesPathArr = imagesDataset.map((image) => {
    const label = image.split(".")[0];
    labels.push(strToLabel(label));
    return `${config.dataSetPath}${image}`;
  });

  const xs = tf.concat(
    imagesPathArr.map((path) => {
      return imgToTensor(path);
    })
  );
  const ys = tf
    .oneHot(tf.tensor(labels).cast("int32"), 36)
    .reshape([imagesDataset.length, 144])
    .cast("float32");

  console.log(xs.shape, ys.shape);

  // 创建模型
  const model = tf.sequential();
  model.add(
    tf.layers.conv2d({
      inputShape: [config.imageHeight, config.imageWidth, 1],
      kernelSize: 3,
      filters: 8,
      strides: 1,
      activation: "relu",
    })
  );
  model.add(
    tf.layers.maxPooling2d({
      poolSize: [2, 2],
      strides: [2, 2],
    })
  );
  model.add(
    tf.layers.conv2d({
      kernelSize: 3,
      filters: 16,
      strides: 1,
      activation: "relu",
    })
  );
  model.add(
    tf.layers.maxPooling2d({
      poolSize: [2, 2],
      strides: [2, 2],
    })
  );
  model.add(
    tf.layers.conv2d({
      kernelSize: 3,
      filters: 32,
      strides: 1,
      activation: "relu",
    })
  );
  model.add(
    tf.layers.maxPooling2d({
      poolSize: [2, 2],
      strides: [2, 2],
    })
  );
  model.add(tf.layers.flatten());
  model.add(
    tf.layers.dense({
      units: 256,
      activation: "relu",
    })
  );
  model.add(
    tf.layers.dense({
      units: 144,
      activation: "softmax",
    })
  );

  // 编译模型
  model.compile({
    optimizer: tf.train.adam(),
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });

  console.log(model.summary());

  try {
    // 训练模型
    await model.fit(xs, ys, {
      batchSize: 32,
      epochs: 10,
      validationSplit: 0.1,
      // @ts-ignore
      onEpochEnd: async (epoch: any, logs: any) => {
        console.log("Epoch: " + epoch + " Loss: " + logs.loss);
      },
    });
  } catch (error) {
    // @ts-ignore
    console.log(error);
    return JSON.stringify(error);
  }

  // 保存模型
  await model.save(config.modelPath);
  return "success";
};

const predict = async (imageBase64: string) => {
  const model = await tf.loadLayersModel(`${config.modelPath}/model.json`);

  const image = imageBase64.replace(/^data:image\/\w+;base64,/, "");
  const imageBuffer = Buffer.from(image, "base64");
  fs.writeFileSync(path.join("./images/test", "test.jpg"), imageBuffer);

  const testImage = imgToTensor("./images/test/test.jpg");
  const result = (await model.predict(testImage)) as tf.Tensor;
  return labelToStr(Array.from(result.dataSync()));
};

export { train, predict };
