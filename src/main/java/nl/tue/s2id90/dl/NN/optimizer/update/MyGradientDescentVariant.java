package nl.tue.s2id90.dl.NN.optimizer.update;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * @author huub
 */
public class MyGradientDescentVariant implements UpdateFunction {
    /**
     * Does a gradient descent step with factor minus learningRate and corrected for batchSize.
     * @param value
     * @param isBias
     * @param gradient
     */
    INDArray update;
    @Override
    public void update(INDArray value, boolean isBias, double learningRate, int batchSize, INDArray gradient) {
        double avgGradient = 0d;
        if (update == null) {
            update = gradient.dup('f').assign(0);
        }
        double factor = -(learningRate/batchSize);
        Nd4j.getBlasWrapper().level1().axpy( value.length(), factor, gradient, value );
                                            // value <-- value + factor * gradient
        /*for (int i = 0; i < gradient.length(); i++) {
            avgGradient += gradient.getDouble(i);
        }
        avgGradient /= gradient.length();
        
        
        for (int i = 0; i < value.length(); i++) {
            value.addi(avgGradient);
        }*/
        
    }
}
