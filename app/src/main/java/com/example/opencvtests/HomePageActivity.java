package com.example.opencvtests;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.view.Window;
import android.widget.ArrayAdapter;
import android.widget.Spinner;

import androidx.appcompat.app.AppCompatActivity;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class HomePageActivity extends AppCompatActivity {

    Spinner spinner;
//    Context context;
    String YOLO_EXTENSION = ".onnx";

    public String[] getYoloVs(){
        try {
            String[] assetContent = this.getAssets().list("");
            int nnCount = 0;
            String[] midContent = new String[assetContent.length];

            for(int i = 0; i < assetContent.length; i++){
                if(assetContent[i].contains(YOLO_EXTENSION)) {
                    midContent[nnCount] = assetContent[i];
                    nnCount++;
                }
                System.out.println(assetContent[i]);
            }

            String[] finalContent = new String[nnCount];
            for(int i = 0; i < nnCount; i++)
                finalContent[i] = midContent[i].replace(YOLO_EXTENSION, "");

            return finalContent;

        } catch (IOException e) {
            System.out.println("No asset data find in this folder");
            e.printStackTrace();
        }

        return new String[0];
    }

    public void onDetectionClick(View Button){
        String YOLOv = spinner.getSelectedItem().toString() + YOLO_EXTENSION;

        Intent camera = new Intent(getApplicationContext(), MainActivity.class);
        camera.putExtra("YOLOv", YOLOv);
        startActivity(camera);
    }

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_home);

        spinner = findViewById(R.id.spinner);
        List<String> YOLOVs = new ArrayList<>();
        YOLOVs.add(0, "Select a YOLO version");

        String[] assetYolos = getYoloVs();
        for (int i = 0; i < assetYolos.length; i++)
        YOLOVs.add(assetYolos[i]);

        ArrayAdapter<String> arrayAdapter = new ArrayAdapter(this, android.R.layout.simple_list_item_1, YOLOVs);
        arrayAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        spinner.setAdapter(arrayAdapter);
    }

    @Override

    protected void onRestart(){

        super.onRestart();//call to restart after onStop

    }

    @Override

    protected void onStart() {

        super.onStart();//soon be visible

    }

    @Override

    protected void onResume() {

        super.onResume();//visible
    }

    @Override

    protected void onPause() {

        super.onPause();//invisible

    }

    @Override

    protected void onStop() {

        super.onStop();

    }

    @Override

    protected void onDestroy() {

        super.onDestroy();

    }

}