/*
 * #%L
 * ImgLib2: a general-purpose, multidimensional image processing library.
 * %%
 * Copyright (C) 2009 - 2012 Stephan Preibisch, Stephan Saalfeld, Tobias
 * Pietzsch, Albert Cardona, Barry DeZonia, Curtis Rueden, Lee Kamentsky, Larry
 * Lindsey, Johannes Schindelin, Christian Dietz, Grant Harris, Jean-Yves
 * Tinevez, Steffen Jaensch, Mark Longair, Nick Perry, and Jan Funke.
 * %%
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 * 
 * The views and conclusions contained in the software and documentation are
 * those of the authors and should not be interpreted as representing official
 * policies, either expressed or implied, of any organization.
 * #L%
 */
package net.imglib2.roi;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import net.imglib2.Cursor;
import net.imglib2.IterableInterval;
import net.imglib2.RealRandomAccess;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgFactory;
import net.imglib2.type.logic.BitType;
import net.imglib2.type.numeric.integer.IntType;

import org.junit.Test;


/**
 * @author Lee Kamentsky
 * 
 * Test cases exercising functionality provided
 * by AbstractIterableRegionOfInterest
 *
 */
public class AbstractIterableRegionOfInterestTest
{
	/**
	 * Regression test of bug 434 - make sure that the interval
	 * returned by getIterableIntervalOverROI does not extend
	 * past the ROI.
	 */
	@Test
	public void testIntervalOfIteratorOverRandomAccessibleInterval() {
		int width = 27;
		int height = 16;
		int depth = 17;
		Img<IntType> img = new ArrayImgFactory<IntType>().create(new long [] {width, height, depth} , new IntType());
		double dimensions [][][] = {
				{ { 1.0, 2.0, 3.0 }, {28.5, 6.0, 7.0 } },
				{ { 1.0, 2.0, 3.0 }, {5.0, 17.5, 7.0 } },
				{ { 1.0, 2.0, 3.0 }, {5.0,  6.0, 18.5 } },
				{ { -1.0, 2.0, 3.0 }, {5.0, 6.0, 7.0 } },
				{ { 1.0, -2.0, 3.0 }, {5.0, 6.0, 7.0 } },
				{ { 1.0, 2.0, -3.0 }, {5.0, 6.0, 7.0 } }};

		for (double [][] dd: dimensions) {
			RectangleRegionOfInterest r = new RectangleRegionOfInterest(dd[0], dd[1]);
			IterableInterval<IntType> ii = r.getIterableIntervalOverROI(img);
			for ( int i = 0; i < ii.numDimensions(); i++ ) {
				assertEquals(Math.max( r.min( i ), img.min(i) ), ii.min( i ));
				assertEquals(Math.min( r.max( i ), img.max(i) ), ii.max( i ));
				assertEquals(Math.max( r.realMin( i ), img.realMin(i) ), ii.realMin( i ), 0);
				assertEquals(Math.min( r.realMax( i ), img.realMax(i) ), ii.realMax( i ), 0);
			}
		}
	}
	/**
	 * Regression test of bug 434 - make sure that the cursor from
	 * getIterableIntervalOverROI().cursor() iterates over the pixels
	 * within a RandomAccessibleInterval that is not entirely within the ROI.
	 * 
	 * Prior to fix, the cursor threw an ArrayIndexOutOfBoundsException.
	 */
	@Test
	public void testCursorOverRandomAccessibleInterval() {
		int width = 27;
		int height = 16;
		int depth = 17;
		Img<IntType> img = new ArrayImgFactory<IntType>().create(new long [] {width, height, depth} , new IntType());
		double dimensions [][][] = {
				{ { 1.0, 2.0, 3.0 }, {28.5, 6.0, 7.0 } },
				{ { 1.0, 2.0, 3.0 }, {5.0, 17.5, 7.0 } },
				{ { 1.0, 2.0, 3.0 }, {5.0,  6.0, 18.5 } },
				{ { -1.0, 2.0, 3.0 }, {5.0, 6.0, 7.0 } },
				{ { 1.0, -2.0, 3.0 }, {5.0, 6.0, 7.0 } },
				{ { 1.0, 2.0, -3.0 }, {5.0, 6.0, 7.0 } }};

		int [] position = new int[img.numDimensions()];
		for (double [][] dd: dimensions) {
			RectangleRegionOfInterest r = new RectangleRegionOfInterest(dd[0], dd[1]);
			IterableInterval<IntType> ii = r.getIterableIntervalOverROI(img);
			boolean mask [][][] = new boolean[width][height][depth];
			RealRandomAccess<BitType> ra = r.realRandomAccess();
			for (int i=0; i<width; i++) {
				ra.setPosition( i, 0);
				for (int j=0; j<height; j++) {
					ra.setPosition(j, 1);
					for (int k=0; k<depth; k++) {
						ra.setPosition(k, 2);
						if (ra.get().get()) mask[i][j][k] = true;
					}
				}
			}
			
			Cursor<IntType> c = r.getIterableIntervalOverROI(img).localizingCursor();
			while(c.hasNext()) {
				c.next();
				c.localize( position );
				assertTrue(mask[position[0]][position[1]][position[2]]);
				mask[position[0]][position[1]][position[2]] = false;
			}
			for (int i=0; i<width; i++) {
				for (int j=0; j<height; j++) {
					for (int k=0; k<depth; k++) {
						assertFalse(mask[i][j][k]);
					}
				}
			}
		}
	}
}
