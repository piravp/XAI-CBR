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

import no.ntnu.mycbr.core.Project;
import no.ntnu.mycbr.core.model.AttributeDesc;
import no.ntnu.mycbr.core.model.SimpleAttDesc;
import no.ntnu.mycbr.core.similarity.config.DistanceConfig;
import no.ntnu.mycbr.core.similarity.config.MultipleConfig;

/**
 * Defines how to compute the similarity of two integers, floats or doubles.
 * The similarity of two numbers is influenced by the range of possible
 * values and their distance.
 * 
 * @author myCBR Team
 *
 */
public abstract class NumberFct extends SimFct {
	
	protected MultipleConfig mc = MultipleConfig.DEFAULT_CONFIG;

	protected double max;
	protected double min;
	protected double diff;

	protected SimpleAttDesc subDesc;

	
	protected DistanceConfig distanceFunction = DistanceConfig.DIFFERENCE;

	
	public NumberFct(Project p, SimpleAttDesc d, String n) {
		super(p,d,n);
		this.subDesc = d;

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
		return isSymmetric;
	}

	/* (non-Javadoc)
	 * @see ISimFct#setMultipleConfig(MultipleConfig)
	 */
	@Override
	public void setMultipleConfig(MultipleConfig mc) {
		this.mc = mc;
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
	public void setSymmetric(boolean symmetric) {
		this.isSymmetric = symmetric; // TODO update
		setChanged();
		notifyObservers();
	}
	
	/**
	 * Sets the mode of this function to mode
	 * @param df the new mode of this function
	 */
	public void setDistanceFct(DistanceConfig df) {
		if (df != distanceFunction) {
			if (df.equals(DistanceConfig.QUOTIENT)) {
				if (min <= 0 && max >= 0) {
					return; // cannot use quotient, when 0 is included in range
				}
			}
			this.distanceFunction = df;
			setChanged();
			notifyObservers();
		}
	}

	/**
	 * Returns the mode of this function
	 * @return the mode of this function
	 */
	public DistanceConfig getDistanceFct() {
		return distanceFunction;
	}

	/* (non-Javadoc)
	 * @see ISimFct#getDescription()
	 */
	@Override
	/**
	 * Returns the description of the attributes which can 
	 * be compared using this function.
	 * @return description this function belongs to
	 */
	public AttributeDesc getDesc() {
		return desc;
	}

	@Override
	public HashMap<String,Object> getRepresentation(){
		HashMap<String,Object> ret = super.getRepresentation();
		ret.put("multipleconfig",MultipleConfig.DEFAULT_CONFIG);
		ret.put("max",this.max);
		ret.put("min",this.min);
		ret.put("diff",this.min);
		ret.put("distancefunction",this.distanceFunction);
		return ret;
	}

}
