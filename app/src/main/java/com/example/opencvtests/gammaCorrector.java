package com.example.opencvtests;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

import java.util.Arrays;
import java.util.List;

public class gammaCorrector {

    boolean gammaCorrection = false;

    public void gammaSwitch() {
        gammaCorrection = !gammaCorrection;
    }

    private double clamp(double val, double min, double max) {
        return Math.max(min, Math.min(max, val));
    }
    private double mean(Mat m){
        double sum = 0;
        for (int j = 0; j < m.rows(); ++j) {
            Mat row = m.row(j);

            for (int i = 0; i < m.cols(); i++) {
                sum += row.get(0, i)[0];
            }
        }
        return sum / (m.rows() * m.cols());
    }
    private Mat pow_clamp_toInt(Mat m, double exp){
        for (int j = 0; j < m.rows(); ++j) {
            Mat row = m.row(j);

            for (int i = 0; i < m.cols(); i++) {
                m.put(j, i, (int) clamp(Math.pow(row.get(0, i)[0], exp), 0,255));
            }
        }
        return m;
    }

    public Mat correctGamma(Mat frame){
        if (gammaCorrection) {
            Mat hsv = frame;
            Imgproc.cvtColor(frame, hsv, Imgproc.COLOR_BGR2HSV);
            List<Mat> hsvLayers = new java.util.ArrayList<Mat>(3);
            Core.split(hsv, hsvLayers);

            Mat hue, sat, val;

            hue = hsvLayers.get(0);
            sat = hsvLayers.get(1);
            val = hsvLayers.get(2);

            double mid = 0.5;
            double mean = mean(val);
            double gamma = Math.log(mid * 255) / Math.log(mean);

            Mat val_gamma = pow_clamp_toInt(val, 1 / gamma);

            List<Mat> hsvRebuilt = Arrays.asList(hue, sat, val_gamma);
            Mat hsv_gamma = hue;
            Core.merge(hsvRebuilt, hsv_gamma);
            Mat frame_gamma = hsv_gamma;
            Imgproc.cvtColor(hsv_gamma, frame_gamma, Imgproc.COLOR_HSV2BGR);

            return frame_gamma;
        }

        else return frame;
    }

}
