/*
 * #%L
 * ImgLib: a general-purpose, multidimensional image processing library.
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

package script.imglib.color.fn;

import java.awt.Color;

import mpicbg.imglib.image.Image;
import mpicbg.imglib.type.numeric.RGBALegacyType;

/** Extracts the HSB saturation of an RGB pixel. */
/**
 * TODO
 *
 */
public abstract class HSBOp extends RGBAOp {

	private final float[] hsb = new float[3];

	public HSBOp(final Image<? extends RGBALegacyType> img) {
		super(img);
	}

	abstract protected int getIndex();

	@Override
	public final double eval() {
		c.fwd();
		final int v = c.getType().get();
		Color.RGBtoHSB((v >> 16) & 0xff, (v >> 8) & 0xff, v & 0xff, hsb);
		return hsb[getIndex()];
	}
}
