package no.ntnu.mycbr.core.retrieval;

import no.ntnu.mycbr.core.ICaseBase;
import no.ntnu.mycbr.core.Project;
import no.ntnu.mycbr.core.casebase.Attribute;
import no.ntnu.mycbr.core.casebase.Instance;
import no.ntnu.mycbr.core.model.AttributeDesc;
import no.ntnu.mycbr.core.similarity.Similarity;
import no.ntnu.mycbr.util.Pair;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.IOException;
import java.util.*;

public class NeuralRetrieval extends RetrievalEngine {


    /**
     *
     */
    private Project prj;

    /**
     *
     */
    private Retrieval retrieval;

    private MultiLayerNetwork mln;

    /**
     *
     * @param p the project this retrieval belongs to
     * @param r the underlying retrieval object
     */
    public NeuralRetrieval(final Project p, final Retrieval r) {
        this.retrieval = r;
        this.prj = p;
        System.out.println("in neuralretrieval");
        //String configFile = "/home/epic/research/dataGetters/operationalmodel_1.0_cat_cross.json";
        //String weightsFile = "/home/epic/research/dataGetters/operationalmodel_1.0_cat_cross.h5";
        String configFile = "/home/epic/research/dataGetters/balanced_operationalmodel.json"; //0.88
        String weightsFile = "/home/epic/research/dataGetters/balanced_operationalmodel.h5";
        mln = null;
        try {
            mln = KerasModelImport.importKerasSequentialModelAndWeights(configFile, weightsFile);
        } catch (InvalidKerasConfigurationException e){
            System.out.println("invalidkerasconf");
        } catch (UnsupportedKerasConfigurationException e){
            System.out.println("unsupportedkerasconf");
        } catch (IOException e ){
            System.out.println("IOexception");
        }


    }

    @Override
    public List<Pair<Instance, Similarity>> retrieve(ICaseBase cb, Instance q) throws Exception {
        Collection<Instance> cases = cb.getCases();
        INDArray q_input = getArray(q);
        double ret = mln.output(q_input).getDouble(0);

        List<Pair<Instance, Similarity>> result = new
                LinkedList<Pair<Instance, Similarity>>();

        for (Instance c : cases) {
            setCurrentCase(c);

            INDArray this_inp = getArray(c);
            double this_ret = mln.output(this_inp).getDouble(0);

            Similarity s = Similarity.get(1.0-Math.abs(this_ret-ret));
            if(s.getValue()==-1.0d){
                System.out.println("ret:"+ret);
                System.out.println("this_ret:"+this_ret);
                System.out.println("(this_ret-ret):"+(this_ret-ret));
            }
            result.add(new Pair<Instance, Similarity>(c,s));
        }
        //System.out.println("result:"+result.size());
        return result;
    }

    INDArray getArray(Instance c) {
        AttributeDesc windspeedattDesc = c.getConcept().getAllAttributeDescs().get("wind_speed");
        Attribute windspeedAttr = c.getAttForDesc(windspeedattDesc);

        AttributeDesc wind_from_direction1_desc = c.getConcept().getAllAttributeDescs().get("wind_from_direction");
        Attribute wind_from_direction_attr = c.getAttForDesc(wind_from_direction1_desc);

        AttributeDesc wind_effect_desc = c.getConcept().getAllAttributeDescs().get("wind_effect");
        Attribute wind_effect_attr = c.getAttForDesc(wind_effect_desc);

        Double windspeed = Double.parseDouble(windspeedAttr.getValueAsString());
        Double wind_from_direction = Double.parseDouble(wind_from_direction_attr.getValueAsString());
        Double windEffect = Double.parseDouble(wind_effect_attr.getValueAsString());
        double[][] arr = new double[1][3];
        arr[0][0] = windspeed;
        arr[0][1] = wind_from_direction;
        arr[0][2] = windEffect;
        INDArray inputarr = Nd4j.create(arr);
        return inputarr;
    }

    @Override
    public List<Pair<Instance, Similarity>> retrieveSorted(ICaseBase cb, Instance q) throws Exception {
        return null;
    }

    @Override
    public List<Pair<Instance, Similarity>> retrieveK(ICaseBase cb, Instance q, int k) throws Exception {
        return null;
    }

    @Override
    public List<Pair<Instance, Similarity>> retrieveKSorted(ICaseBase cb, Instance q, int k) throws Exception {
        return null;
    }
}
