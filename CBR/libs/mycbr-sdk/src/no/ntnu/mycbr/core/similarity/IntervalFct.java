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

package no.ntnu.mycbr.core.similarity;

import java.util.HashMap;
import java.util.Observable;

import no.ntnu.mycbr.core.Project;
import no.ntnu.mycbr.core.casebase.Attribute;
import no.ntnu.mycbr.core.casebase.IntervalAttribute;
import no.ntnu.mycbr.core.casebase.MultipleAttribute;
import no.ntnu.mycbr.core.casebase.SpecialAttribute;
import no.ntnu.mycbr.core.model.AttributeDesc;
import no.ntnu.mycbr.core.model.IntervalDesc;
import no.ntnu.mycbr.core.similarity.config.MultipleConfig;

/**
 * Not implemented yet.
 * 
 * @author myCBR Team
 *
 */
public class IntervalFct extends SimFct {

	private IntervalDesc subDesc;
	protected MultipleConfig mc = MultipleConfig.DEFAULT_CONFIG;
	
	public IntervalFct(Project prj,IntervalDesc desc2, String name) {
		super(prj,desc2,name);
		this.subDesc = desc2;
	}
	
	/* (non-Javadoc)
	 * @see ISimFct#calculateSimilarity(Attribute, Attribute)
	 */
	@Override
	public Similarity calculateSimilarity(Attribute value1, Attribute value2)
			throws Exception {
		if (value1 instanceof SpecialAttribute || value2 instanceof SpecialAttribute) {
			return prj.calculateSpecialSimilarity(value1, value2);
		} else if (value1 instanceof MultipleAttribute<?> && value2 instanceof MultipleAttribute<?>) {
			return prj.calculateMultipleAttributeSimilarity(this, (MultipleAttribute<?>)value1, (MultipleAttribute<?>)value2);
		} else if (value1 instanceof IntervalAttribute && value2 instanceof IntervalAttribute) {
			IntervalAttribute v1 = (IntervalAttribute) value1;
			IntervalAttribute v2 = (IntervalAttribute) value2;
			if (v1.equals(v2)) {
				return Similarity.get(1.0);
			} 
			return Similarity.get(0.0);
		} else {
			return Similarity.INVALID_SIM;
		}
		
	}

	/* (non-Javadoc)
	 * @see ISimFct#getDesc()
	 */
	@Override
	public AttributeDesc getDesc() {
		return desc;
	}

	/* (non-Javadoc)
	 * @see ISimFct#getMultipleConfig()
	 */
	@Override
	public MultipleConfig getMultipleConfig() {
		return mc;
	}

	/* (non-Javadoc)
	 * @see ISimFct#getName()
	 */
	@Override
	public String getName() {
		return name;
	}

	/* (non-Javadoc)
	 * @see ISimFct#getProject()
	 */
	@Override
	public Project getProject() {
		return prj;
	}

	/* (non-Javadoc)
	 * @see ISimFct#isSymmetric()
	 */
	@Override
	public boolean isSymmetric() {
		return true;
	}
	
	/* (non-Javadoc)
	 * @see ISimFct#setMultipleConfig(MultipleConfig)
	 */
	@Override
	public void setMultipleConfig(MultipleConfig mc) {
		this.mc = mc;
		setChanged();
		notifyObservers();
	}

	/* (non-Javadoc)
	 * @see ISimFct#setName(java.lang.String)
	 */
	@Override
	/**
	 * Sets the name of this function to name.
	 * Does nothing if there is another function with this name.
	 * @param name the name of this function
	 */
	public void setName(String name) {
		if (subDesc.getFct(name) == null) {
			subDesc.renameFct(this.name, name);
			this.name = name;
			setChanged();
			notifyObservers();
		}
	}

	/* (non-Javadoc)
	 * @see ISimFct#setSymmetric(boolean)
	 */
	@Override
	public void setSymmetric(boolean symmetric) {}

	/* (non-Javadoc)
	 * @see java.util.Observer#update(java.util.Observable, java.lang.Object)
	 */
	@Override
	public void update(Observable o, Object arg) {
		setChanged();
		notifyObservers();
	}

	/* (non-Javadoc)
	 * @see ISimFct#clone(AttributeDesc, boolean)
	 */
	@Override
	public void clone(AttributeDesc descNEW, boolean active) {
		if (descNEW instanceof IntervalDesc && !name.equals(Project.DEFAULT_FCT_NAME)) {
			IntervalFct f = ((IntervalDesc)descNEW).addIntervalFct(name, active);
			f.mc = this.mc;
		}
	}
	@Override
	public HashMap<String, Object> getRepresentation() {
		HashMap<String,Object> ret = super.getRepresentation();
		ret.put("type",this.getClass().getName());
		ret.put("multipleConfig",this.mc);
		return ret;
	}
}
