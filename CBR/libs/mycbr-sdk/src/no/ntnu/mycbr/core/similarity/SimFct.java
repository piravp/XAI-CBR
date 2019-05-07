package no.ntnu.mycbr.core.similarity;

import no.ntnu.mycbr.core.Project;
import no.ntnu.mycbr.core.casebase.Attribute;
import no.ntnu.mycbr.core.model.AttributeDesc;
import no.ntnu.mycbr.core.similarity.config.MultipleConfig;

import java.util.HashMap;
import java.util.Observable;

public class SimFct extends Observable implements ISimFct {
    protected String name;

    /**
     * The description of the given attributes
     */
    protected AttributeDesc desc;

    protected boolean isSymmetric = true;
    protected Project prj;

    public SimFct(Project p, AttributeDesc desc, String name){
        this.name = name;
        this.prj = p;
        this.desc = desc;
        desc.addObserver(this);
    }
    @Override
    public Similarity calculateSimilarity(Attribute value1, Attribute value2) throws Exception {
        return null;
    }

    @Override
    public boolean isSymmetric() {
        return false;
    }

    @Override
    public String getName() {
        return null;
    }

    @Override
    public void setName(String name) {

    }

    @Override
    public AttributeDesc getDesc() {
        return null;
    }

    @Override
    public void setSymmetric(boolean symmetric) {

    }

    @Override
    public MultipleConfig getMultipleConfig() {
        return null;
    }

    @Override
    public void setMultipleConfig(MultipleConfig mc) {

    }

    @Override
    public Project getProject() {
        return null;
    }

    @Override
    public void clone(AttributeDesc descNEW, boolean active) {

    }

    @Override
    public HashMap<String, Object> getRepresentation() {
        HashMap<String,Object> ret = new HashMap<>();
        ret.put("name",name);
        ret.put("isSymmetric",isSymmetric);
        ret.put("attribute",desc.getName());
        ret.put("project",prj.getName());
        return ret;
    }

    @Override
    public void update(Observable observable, Object o) {

    }
}
