package com.example.opencvtests;

import static org.opencv.imgproc.Imgproc.FONT_HERSHEY_SIMPLEX;

import android.content.Context;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfRect2d;
import org.opencv.core.Point;
import org.opencv.core.Rect2d;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgproc.Imgproc;
import org.opencv.utils.Converters;

import java.io.BufferedInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class yoloDetector {
    String yoloV;
    boolean startedYolo = false;
    boolean firstTimeYolo = false;
    Net yolo;
    Context context;
    List<String> labels = Arrays.asList("ebi maki", "ebi nigiri", "unagi nigiri", "maguro maki", "maguro nigiri", "sake maki", "sake nigiri", "suzuki nigiri", "tako nigiri", "edamame", "wakame", "gyoza", "shao mai", "tempura", "temaki");

    int INPUT_WIDTH = 640;
    int INPUT_HEIGHT = 640;
    float SCORE_THRESHOLD = 0.2f;
    float NMS_THRESHOLD = 0.2f;
    float CONFIDENCE_THRESHOLD = 0.4f;

    public yoloDetector(Context activity, String yoloPath) {
        context = activity;

        yoloV = yoloPath;


    }

    public MatOfByte getYolo() {
        InputStream inputStream;
        MatOfByte weights = new MatOfByte();

        try {
            inputStream = new BufferedInputStream(context.getAssets().open(yoloV));
            byte[] data = new byte[inputStream.available()];
            inputStream.read(data);
            inputStream.close();
            weights.fromArray(data);
        } catch (IOException e) {
            e.printStackTrace();
        }

        return weights;
    }

    public void startYolo(){
        if (startedYolo == false) {
            startedYolo = true;

            if (firstTimeYolo == false) {
                firstTimeYolo = true;

                MatOfByte yo = getYolo();

                yolo = Dnn.readNetFromONNX(yo);

                try {
                    Thread.sleep(200);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
        } else {
            startedYolo = false;
        }
    }

    public void resumeYolo(){
        if (startedYolo == true) {
            MatOfByte yo = getYolo();

            yolo = Dnn.readNetFromONNX(yo);

            try {
                Thread.sleep(200);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }

    public Mat detect(Mat frame) {
        Mat imageBlob = Dnn.blobFromImage(frame, 1 / 255.0, new Size(INPUT_WIDTH, INPUT_HEIGHT), new Scalar(0, 0, 0),/*swapRB*/false, /*crop*/false);

        yolo.setInput(imageBlob);

        java.util.List<Mat> result = new java.util.ArrayList<Mat>(3);

        yolo.forward(result, yolo.getUnconnectedOutLayersNames());

        Mat out = result.get(0);

        int sz[] = {out.size(1), out.size(2)};
        Mat data = out.reshape(1, sz);

        return data;
    }

    public Mat getAndDrawBoundingBoxes(Mat frame, Mat data) {
        float x_factor = (float) frame.cols() / INPUT_WIDTH;
        float y_factor = (float) frame.rows() / INPUT_HEIGHT;

        List<Integer> clsIds = new ArrayList<>();
        List<Float> confs = new ArrayList<>();
        List<Rect2d> rects = new ArrayList<>();

        for (int j = 0; j < data.rows(); ++j) {
            Mat row = data.row(j);

            float confidence = (float) row.get(0, 4)[0];

            if (confidence > CONFIDENCE_THRESHOLD) {
                Mat scores = row.colRange(5, data.cols());
                Core.MinMaxLocResult mm = Core.minMaxLoc(scores);
                if (mm.maxVal > SCORE_THRESHOLD) {
                    Point classIdPoint = mm.maxLoc;

                    float x = (float) row.get(0, 0)[0];
                    float y = (float) row.get(0, 1)[0];
                    float w = (float) row.get(0, 2)[0];
                    float h = (float) row.get(0, 3)[0];

                    int left = (int) ((x - 0.5 * w) * x_factor);
                    int top = (int) ((y - 0.5 * h) * y_factor);
                    int width = (int) (w * x_factor);
                    int height = (int) (h * y_factor);

                    clsIds.add((int) classIdPoint.x);
                    confs.add((float) confidence);

                    rects.add(new Rect2d(left, top, width, height));
                }
            }
        }

        int ArrayLength = confs.size();

        if (ArrayLength >= 1) {
            // Apply non-maximum suppression procedure.

            MatOfFloat confidences = new MatOfFloat(Converters.vector_float_to_Mat(confs));

            Rect2d[] boxesArray = rects.toArray(new Rect2d[0]);

            MatOfRect2d boxes = new MatOfRect2d(boxesArray);

            MatOfInt indices = new MatOfInt();

            Dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD, indices);

            int[] ind = indices.toArray();

            Mat labelledFrame = setLabels(frame, clsIds, confs, boxesArray, ind);
            return labelledFrame;
        }

        else return frame;
    }

    private Mat setLabels(Mat frame, List<Integer> clsIds, List<Float> confs, Rect2d[] boxesArray, int[] ind) {
        for (int i = 0; i < ind.length; ++i) {

            int idx = ind[i];
            Rect2d box = boxesArray[idx];

            int idGuy = clsIds.get(idx);

            float conf = confs.get(idx);

            int intConf = (int) (conf * 100);

            Imgproc.putText(frame, labels.get(idGuy) + " " + intConf + "%", box.tl(), FONT_HERSHEY_SIMPLEX, 0.5, new Scalar(255, 255, 0), 1);

            Imgproc.rectangle(frame, box.tl(), box.br(), new Scalar(255, 0, 0), 2);
        }

        return frame;
    }
}
