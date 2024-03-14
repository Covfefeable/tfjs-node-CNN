import Express from "express";
import cookieParser from "cookie-parser";
import bodyParser from "body-parser";
import path from "path";

import * as tf from "@tensorflow/tfjs-node";
import fs from "fs";

import { predict, train } from "./src/train";
import { imgToTensor, labelToStr } from "./src/const";
import axios from "axios";

let app = Express();
let PORT = 3000;

app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: false }));
app.use(cookieParser());

app.get("/train", async (req, res) => {
  const r = await train();
  res.send(r);
});

app.get("/predict", async (req, res) => {
  // 抓取一张图片并使用模型预测
  const url = req.query.url as string;
  const { data } = await axios.get(url, {
    responseType: "arraybuffer",
  });
  const base64 = Buffer.from(data, "binary").toString("base64");
  const result = await predict(base64);
  res.send(result);
});

app.set("port", PORT || 3000);
app.listen(app.get("port"), function () {
  console.log("Express server listening...");
});
