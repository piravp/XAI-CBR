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

import java.util.*;

import no.ntnu.mycbr.core.Project;
import no.ntnu.mycbr.core.casebase.*;
import no.ntnu.mycbr.core.casebase.Attribute;
import no.ntnu.mycbr.core.casebase.DateAttribute;
import no.ntnu.mycbr.core.casebase.MultipleAttribute;
import no.ntnu.mycbr.core.casebase.SpecialAttribute;
import no.ntnu.mycbr.core.model.AttributeDesc;
import no.ntnu.mycbr.core.model.DateDesc;
import no.ntnu.mycbr.core.similarity.config.MultipleConfig;

/**
 * Not implemented yet.
 * 
 * @author myCBR Team
 *
 */
public class DateFct extends SimFct {

    public enum DateFunctionPrecision { Second, Minute, Hour, Day, Month, Year }

	private DateDesc subDesc;
	protected MultipleConfig mc = MultipleConfig.DEFAULT_CONFIG;
    private DateFunctionPrecision precision;

	public DateFct(Project prj,DateDesc desc, String name, DateFunctionPrecision precision) {
		super(prj,desc,name);
		this.subDesc = desc;
        this.precision = precision;
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
		} else if (value1 instanceof DateAttribute && value2 instanceof DateAttribute) {

            DateAttribute dateAttr1 = (DateAttribute) value1;
            DateAttribute dateAttr2 = (DateAttribute) value2;
            Date date1 = dateAttr1.getDate();
            Date date2 = dateAttr2.getDate();
            double result = 0.0;
            Calendar calendar1 = GregorianCalendar.getInstance();
            calendar1.setTime(date1);
            Calendar calendar2 = GregorianCalendar.getInstance();
            calendar2.setTime(date2);

            switch (this.precision) {
                case Second:
                    if (calendar1.get(Calendar.MINUTE) == calendar2.get(Calendar.MINUTE)
                            && calendar1.get(Calendar.HOUR_OF_DAY) == calendar2.get(Calendar.HOUR_OF_DAY)
                            && calendar1.get(Calendar.DAY_OF_MONTH) == calendar2.get(Calendar.DAY_OF_MONTH)
                            && calendar1.get(Calendar.MONTH) == calendar2.get(Calendar.MONTH)
                            && calendar1.get(Calendar.YEAR) == calendar2.get(Calendar.YEAR)) {
                        int secondsDate1 = calendar1.get(Calendar.SECOND);
                        int secondsDate2 = calendar2.get(Calendar.SECOND);
                        result = 1 - (Math.abs(secondsDate2 - secondsDate1) / 60.0);
                    }
                    break;
                case Minute:
                    if (calendar1.get(Calendar.HOUR_OF_DAY) == calendar2.get(Calendar.HOUR_OF_DAY)
                            && calendar1.get(Calendar.DAY_OF_MONTH) == calendar2.get(Calendar.DAY_OF_MONTH)
                            && calendar1.get(Calendar.MONTH) == calendar2.get(Calendar.MONTH)
                            && calendar1.get(Calendar.YEAR) == calendar2.get(Calendar.YEAR)) {
                        int minutesDate1 = calendar1.get(Calendar.MINUTE);
                        int minutesDate2 = calendar2.get(Calendar.MINUTE);
                        result = 1 - (Math.abs(minutesDate2 - minutesDate1) / 60.0);
                    }
                    break;
                case Hour:
                    if (calendar1.get(Calendar.DAY_OF_MONTH) == calendar2.get(Calendar.DAY_OF_MONTH)
                            && calendar1.get(Calendar.MONTH) == calendar2.get(Calendar.MONTH)
                            && calendar1.get(Calendar.YEAR) == calendar2.get(Calendar.YEAR)) {
                        int hoursDate1 = calendar1.get(Calendar.HOUR_OF_DAY);
                        int hoursDate2 = calendar2.get(Calendar.HOUR_OF_DAY);
                        result = 1 - (Math.abs(hoursDate2 - hoursDate1) / 24.0);
                    }
                    break;
                case Day:
                    if (calendar1.get(Calendar.MONTH) == calendar2.get(Calendar.MONTH)
                            && calendar1.get(Calendar.YEAR) == calendar2.get(Calendar.YEAR)) {
                        int daysDate1 = calendar1.get(Calendar.DAY_OF_MONTH);
                        int daysDate2 = calendar2.get(Calendar.DAY_OF_MONTH);
                        result = 1 - (Math.abs(daysDate2 - daysDate1) / 31.0); // Muss bzgl. Schaltjahr angepasst werden
                    }
                    break;
                case Month:
                    if (calendar1.get(Calendar.YEAR) == calendar2.get(Calendar.YEAR)) {
                        int monthsDate1 = calendar1.get(Calendar.MONTH);
                        int monthsDate2 = calendar2.get(Calendar.MONTH);
                        result = 1 - (Math.abs(monthsDate2 - monthsDate1) / 12.0);
                    }
                    break;
                case Year:
                    int yearsDate1 = calendar1.get(Calendar.YEAR);
                    int yearsDate2 = calendar2.get(Calendar.YEAR);
                    result = 1 - (Math.abs(yearsDate2 - yearsDate1) / 100.0); // Konfigurierbar machen
                    break;
            }

            return  Similarity.get(result);

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
	 */
    public DateFunctionPrecision getPrecision() {
        return precision;
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
	public void update(Observable o, Object arg) {}

	/* (non-Javadoc)
	 * @see ISimFct#clone(AttributeDesc, boolean)
	 */
	@Override
	public void clone(AttributeDesc descNEW, boolean active) {
		if (descNEW instanceof DateDesc && !name.equals(Project.DEFAULT_FCT_NAME)) {
			DateFct f = ((DateDesc)descNEW).addDateFct(name, active, this.precision);
			f.mc = this.mc;
		}
	}

    public void setPrecision(DateFunctionPrecision adj) {
        this.precision = adj;
    }

	@Override
	public HashMap<String,Object> getRepresentation(){
		HashMap<String,Object> ret = super.getRepresentation();
		ret.put("type",this.getClass().getName());
		ret.put("precision",this.precision);
		ret.put("multipleConfig",this.mc);
		return ret;
	}
}
