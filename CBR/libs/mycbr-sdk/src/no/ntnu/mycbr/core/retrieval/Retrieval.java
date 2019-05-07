/*
 * myCBR License 3.0
 * 
 * Copyright (c) 2006-2015, by German Research Center for Artificial Intelligence (DFKI GmbH), Germany
 * 
 * Project Website: http://www.mycbr-project.net/
 * 
 * This library is free software; you can redistribute it and/or modify 
 * it under the terms of the GNU Lesser General Public License as published by 
 * the Free Software Foundation; either version 3 of the License, or 
 * (at your option) any later version.
 * 
 * This library is distributed in the hope that it will be useful, 
 * but WITHOUT ANY WARRANTY; without even the implied warranty of 
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
 * See the GNU Lesser General Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public License 
 * along with this library; if not, write to the Free Software Foundation, Inc., 
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 * 
 * Oracle and Java are registered trademarks of Oracle and/or its affiliates. 
 * Other names may be trademarks of their respective owners.
 * 
 * endOfLic */

package no.ntnu.mycbr.core.retrieval;

import java.util.HashMap;
import java.util.List;
import java.util.concurrent.Callable;


import no.ntnu.mycbr.core.ICaseBase;
import no.ntnu.mycbr.core.Project;
import no.ntnu.mycbr.core.casebase.Instance;
import no.ntnu.mycbr.core.model.Concept;
import no.ntnu.mycbr.core.similarity.Similarity;
import no.ntnu.mycbr.util.Pair;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

/**
 * A retrieval has a retrieval method and a retrieval engine.
 * When specifying a case base and a query you can retrieve a similarity
 * value between the query and each case in the case base.
 * 
 * @author myCBR Team
 *
 */
public class Retrieval extends HashMap<Instance, Similarity> implements Callable<RetrievalResult> {

    private final Log logger = LogFactory.getLog(getClass());

    protected RetrievalMethod retrievalMethod = RetrievalMethod.RETRIEVE;
    //List<Pair<Instance,Similarity>> l = null;
    /**
     *
     */
    private static final long serialVersionUID = 2656679620557431799L;

    /**
     *
     */
    private Instance query;

    /**
     *
     */
    private RetrievalEngine re;

    /**
     *
     */
    private Project p;

    private ICaseBase cb;
    private int k = 5;
	private boolean finished = true;

    @Override
    public RetrievalResult call() throws Exception {
        List<Pair<Instance,Similarity>> l = null;
        finished = false;
        l = getResults(l);
        return new RetrievalResult(this.retrievalID,l);
    }

    public String getRetrievalID() {
        return retrievalID;
    }

    public void setRetrievalID(String retrievalID) {
        this.retrievalID = retrievalID;
    }

    public interface RetrievalCustomer{
	    public void addResults(Retrieval ret, List<Pair<Instance,Similarity>> results);
    }
    private RetrievalCustomer customer;
    
    /**
     *
     * @param c the query should be an instance of this concept
     */
    public Retrieval(final Concept c, ICaseBase cb, RetrievalCustomer rc) {
        p = c.getProject();
        this.cb = cb;
        query = c.getQueryInstance();
        re = new SequentialRetrieval(p, this);
        //re = new NeuralRetrieval(p, this);
        this.customer = rc;
    }
    private String retrievalID = null;
    /**
     * The retrieval is based on a case ID
     * @param c
     */
    public Retrieval(final Concept c, ICaseBase cb, RetrievalCustomer rc, String retrievalID) {
        p = c.getProject();
        this.cb = cb;
        query = c.getQueryInstance();
        re = new SequentialRetrieval(p, this);
        //re = new NeuralRetrieval(p, this);
        this.customer = rc;
        this.retrievalID = retrievalID;
    }

    /**
    *
    * @param c the query should be an instance of this concept
    */
   public Retrieval(final Concept c, ICaseBase cb, RetrievalEngine re, RetrievalCustomer rc) {
       p = c.getProject();
       this.cb = cb;
       query = c.getQueryInstance();
       this.re = re;
       this.customer = rc;
   }
   
    /**
     *
     * @throws Exception if something goes wrong during retrieval
     */
    public final void start() {
    	run();
    }
    
    /**
     * @since 3.0.0 BETA 0.2
     */
    public void setRetrievalEngine(RetrievalEngine re) {
    	this.re = re;
    }
    
    /**
     * 
     */
    public RetrievalEngine getRetrievalEngine() {
    	return re;
    }
    
    /**
     * @since 3.0.0 BETA 0.3
     */
    public void setCaseBase(ICaseBase cb) {
    	this.cb = cb;
    }
    
    /**
     * @since 3.0.0 BETA 0.3
     */
    public ICaseBase getCaseBase() {
    	return cb;
    }
    
    /**
     * @since 3.0.0 BETA 0.2
     */
    public Instance getQueryInstance() {
    	return query;
    }
    
    /**
     * Set all attributes to undefined
     * @since 3.0.0 BETA 0.3
     */
    public Instance resetQuery() {
    	query.setAttsUnknown();
		return query;
    }
    
    /**
     * @since 3.0.0 BETA 0.3
     */
    public void setK(int k) {
    	this.k = k;
    }
    
    /**
     * @since 3.0.0 BETA 0.3
     */
    public int getK() {
    	return k;
    }
    
    /**
    *
    * @param m the current retrieval method
    */
    public final void setRetrievalMethod(RetrievalMethod m) {
       retrievalMethod = m;
   }

    public boolean isFinished() {
    	return finished;
    }
   /**
    *
    * @return the current case
    */
    public final RetrievalMethod getRetrievalMethod() {
       return retrievalMethod;
   }
   
    public enum RetrievalMethod {
    	RETRIEVE, RETRIEVE_SORTED, RETRIEVE_K, RETRIEVE_K_SORTED;
    }

	/* (non-Javadoc)
	 * @see java.lang.Runnable#run()
	 */
	//@Override
	public void run() {
        List<Pair<Instance,Similarity>> l = null;
		finished = false;
    	logger.debug("starting retrieval in thread run(): ");
        l = getResults(l);
        this.customer.addResults(this,l);
        //logger.info("ending retrieval hashcode: "+this.hashCode());
    	finished = true;
	}

    private List<Pair<Instance, Similarity>> getResults(List<Pair<Instance, Similarity>> l) {
        try {
            // start new retrieval
            switch(retrievalMethod) {
                case RETRIEVE: 			l = re.retrieve(cb, query);
                                        break;

                case RETRIEVE_SORTED: 	l = re.retrieveSorted(cb, query);
                                        break;

                case RETRIEVE_K: 		l = re.retrieveK(cb, query, k);
                                        break;

                case RETRIEVE_K_SORTED: l = re.retrieveKSorted(cb, query, k);
                                        break;

                default: 				l = re.retrieve(cb, query);
                                         break;
            }

        } catch(Exception e) {
            //System.out.println("Retrieval");
            e.printStackTrace();
        }
        return l;
    }

}
