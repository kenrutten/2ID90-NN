/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package nl.tue.s2id90.dl.NN.transform;

import java.util.List;
import nl.tue.s2id90.dl.NN.tensor.TensorPair;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 *
 * @author s169713
 */
public class MeanSubtraction implements DataTransform {
    Double mean = 0d;
    int nPixels = 0; //number of pixels used to calculate the mean
    @Override public void fit(List<TensorPair> data) {
        if (data.isEmpty()) {
            throw new IllegalArgumentException("Empty dataset");
        }
        for(TensorPair pair : data) {
            //System.out.println(pair);
            INDArray ind = pair.model_input.getValues();
            for (int i = 0; i < 28; i++) {
                for (int j = 0; j < 28; j++) {
                    mean += ind.getDouble(1,1,i,j);
                    nPixels++;
                }
            }
        }
        mean /= nPixels;
        System.out.println("Mean: " + mean);
    }
    @Override public void transform(List<TensorPair> data) {
        for (TensorPair pair : data) {
            INDArray ind = pair.model_input.getValues();
            ind.subi(mean);
            //System.out.println("PAIR: " + pair);
        }
    }
    
}
