package nl.tue.s2id90.dl.NN.optimizer.update;

import java.util.function.Supplier;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * @author huub
 */
public class L2Decay implements UpdateFunction {
    double decay;
    UpdateFunction f;
    public L2Decay(Supplier<UpdateFunction> supplier, double decay) {
        this.decay = decay;
        this.f = supplier.get();
    }
    @Override
    public void update(INDArray value, boolean isBias, double learningRate, int batchSize, INDArray gradient) {
        if (!isBias) {
            value.muli(decay);
        }
        f.update(value, isBias, learningRate, batchSize, gradient);
    }
}
