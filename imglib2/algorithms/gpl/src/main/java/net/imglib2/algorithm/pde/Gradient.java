/*
 * #%L
 * ImgLib2: a general-purpose, multidimensional image processing library.
 * %%
 * Copyright (C) 2009 - 2012 Stephan Preibisch, Stephan Saalfeld, Tobias
 * Pietzsch, Albert Cardona, Barry DeZonia, Curtis Rueden, Lee Kamentsky, Larry
 * Lindsey, Johannes Schindelin, Christian Dietz, Grant Harris, Jean-Yves
 * Tinevez, Steffen Jaensch, Mark Longair, Nick Perry, and Jan Funke.
 * %%
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as
 * published by the Free Software Foundation, either version 2 of the 
 * License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public 
 * License along with this program.  If not, see
 * <http://www.gnu.org/licenses/gpl-2.0.html>.
 * #L%
 */
package net.imglib2.algorithm.pde;

import net.imglib2.Cursor;
import net.imglib2.RandomAccess;
import net.imglib2.algorithm.MultiThreadedBenchmarkAlgorithm;
import net.imglib2.algorithm.OutputAlgorithm;
import net.imglib2.exception.IncompatibleTypeException;
import net.imglib2.img.Img;
import net.imglib2.outofbounds.OutOfBounds;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.view.Views;

public class Gradient<T extends RealType<T>> extends MultiThreadedBenchmarkAlgorithm implements OutputAlgorithm<Img<FloatType>> {

	private final Img<T> input;
	private Img<FloatType> output;
	private final boolean[] doDimension;

	/*
	 * CONSTRUCTOR	
	 */

	public Gradient(final Img<T> input, final boolean[] doDimension) {
		this.input = input;
		this.doDimension = doDimension;
		long[] dimensions = new long[input.numDimensions()+1];
		for (int i = 0; i < dimensions.length-1; i++) {
			dimensions[i] = input.dimension(i);
		}
		dimensions[dimensions.length-1] = input.numDimensions();
		try {
			this.output = input.factory().imgFactory(new FloatType()).create(dimensions , new FloatType());
		} catch (IncompatibleTypeException e) {
			e.printStackTrace();
		}
	}
	
	
	@Override
	public boolean checkInput() {
		return true;
	}

	@Override
	public boolean process() {
		
		long start = System.currentTimeMillis();
		
		Cursor<T> in = input.localizingCursor();
		RandomAccess<FloatType> oc = output.randomAccess();
		T zero = input.firstElement().createVariable();
		OutOfBounds<T> ra = Views.extendValue(input, zero).randomAccess();
		
		float central, diff;
		
		int newdim = input.numDimensions();
		
		while (in.hasNext()) {
			in.fwd();

			// Position neighborhood cursor;
			ra.setPosition(in);
			
			// Position output cursor
			for (int i = 0; i < input.numDimensions(); i++) {
				oc.setPosition(in.getLongPosition(i), i);
			}
			oc.setPosition(0, newdim);
			
			// Central value
			central =  in.get().getRealFloat();

			// Gradient
			for (int i = 0; i < input.numDimensions(); i++) {
				if (!doDimension[i]) {
					continue;
				}
				ra.fwd(i);
				diff = central - ra.get().getRealFloat();
				ra.bck(i);
				
				oc.get().set(diff);
				oc.fwd(newdim);
			}
			
		}
		
		processingTime = System.currentTimeMillis() - start;
		return true;
	}


	@Override
	public Img<FloatType> getResult() {
		return output;
	}

}
