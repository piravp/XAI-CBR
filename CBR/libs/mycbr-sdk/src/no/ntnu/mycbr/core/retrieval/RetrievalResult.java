package no.ntnu.mycbr.core.retrieval;

//This class has to be used when doing async/parallell retrieval
//

import no.ntnu.mycbr.core.casebase.Instance;
import no.ntnu.mycbr.core.similarity.Similarity;
import no.ntnu.mycbr.util.Pair;

import java.util.List;

public class RetrievalResult {
    private String retrevalID;
    private List<Pair<Instance,Similarity>> result;
    public RetrievalResult(String retrievalID, List<Pair<Instance,Similarity>> result){
        this.setRetrevalID(retrievalID);
        this.setResult(result);
    }

    public String getRetrevalID() {
        return retrevalID;
    }

    public void setRetrevalID(String retrevalID) {
        this.retrevalID = retrevalID;
    }

    public List<Pair<Instance, Similarity>> getResult() {
        return result;
    }

    public void setResult(List<Pair<Instance, Similarity>> result) {
        this.result = result;
    }
}
