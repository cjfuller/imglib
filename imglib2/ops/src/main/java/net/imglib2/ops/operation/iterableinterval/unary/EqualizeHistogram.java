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

package net.imglib2.ops.operation.iterableinterval.unary;

import java.util.Random;

import net.imglib2.Cursor;
import net.imglib2.IterableInterval;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgFactory;
import net.imglib2.ops.operation.UnaryOperation;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;

/**
 * TODO
 * 
 * @author Martin Horn (University of Konstanz)
 */
public class EqualizeHistogram<T extends RealType<T>> implements
        UnaryOperation<IterableInterval<T>, IterableInterval<T>> {

    private final int numBins;

    public EqualizeHistogram(int numBins) {
        this.numBins = numBins;

    }

    @Override
    public IterableInterval<T> compute(IterableInterval<T> in,
            IterableInterval<T> r) {

        assert (in.iterationOrder().equals(r.iterationOrder()));

        OpsHistogram histo = new MakeHistogram<T>(numBins).compute(in);
        T val = r.firstElement().createVariable();

        int min = (int)val.getMaxValue();
        // calc cumulated histogram
        for (int i = 1; i < histo.numBins(); i++) {
            histo.hist()[i] = histo.get(i) + histo.get(i - 1);
            if (histo.get(i) != 0) {
                min = Math.min(min, histo.get(i));
            }
        }

        double gmax = histo.numBins();

        Cursor<T> cin = in.cursor();
        Cursor<T> cout = r.cursor();

        long numPix = r.size();

        while (cin.hasNext()) {
            cin.fwd();
            cout.fwd();

            val = cin.get();
            int p = histo.getByValue(val.getRealFloat());
            double t = (p - min);
            t /= numPix - min;
            t *= gmax;
            p = (int)Math.round(t);
            cout.get().setReal(histo.binToValue(p));
        }
        return r;

    }

    @Override
    public UnaryOperation<IterableInterval<T>, IterableInterval<T>> copy() {
        return new EqualizeHistogram<T>(numBins);
    }

    public static void main(String[] args) {
        Img<FloatType> test =
                new ArrayImgFactory<FloatType>().create(new int[]{10, 10},
                        new FloatType());
        Random rand = new Random();
        for (FloatType t : test) {
            t.setReal(rand.nextDouble() * Float.MAX_VALUE);
        }
        new EqualizeHistogram<FloatType>(256).compute(test, test);
    }
}
