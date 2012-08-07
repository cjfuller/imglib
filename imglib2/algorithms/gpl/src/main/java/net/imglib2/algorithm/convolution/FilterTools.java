/*
 * ------------------------------------------------------------------------
 *
 *  Copyright (C) 2003 - 2010
 *  University of Konstanz, Germany and
 *  KNIME GmbH, Konstanz, Germany
 *  Website: http://www.knime.org; Email: contact@knime.org
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License, Version 3, as
 *  published by the Free Software Foundation.
 *
 *  This program is distributed in the hope that it will be useful, but
 *  WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, see <http://www.gnu.org/licenses>.
 *
 *  Additional permission under GNU GPL version 3 section 7:
 *
 *  KNIME interoperates with ECLIPSE solely via ECLIPSE's plug-in APIs.
 *  Hence, KNIME and ECLIPSE are both independent programs and are not
 *  derived from each other. Should, however, the interpretation of the
 *  GNU GPL Version 3 ("License") under any applicable laws result in
 *  KNIME and ECLIPSE being a combined program, KNIME GMBH herewith grants
 *  you the additional permission to use and propagate KNIME together with
 *  ECLIPSE with only the license terms in place for ECLIPSE applying to
 *  ECLIPSE and the GNU GPL Version 3 applying for KNIME, provided the
 *  license terms of ECLIPSE themselves allow for the respective use and
 *  propagation of ECLIPSE together with KNIME.
 *
 *  Additional permission relating to nodes for KNIME that extend the Node
 *  Extension (and in particular that are based on subclasses of NodeModel,
 *  NodeDialog, and NodeView) and that only interoperate with KNIME through
 *  standard APIs ("Nodes"):
 *  Nodes are deemed to be separate and independent programs and to not be
 *  covered works.  Notwithstanding anything to the contrary in the
 *  License, the License does not apply to Nodes, you are not required to
 *  license Nodes under the License, and you are granted a license to
 *  prepare and propagate Nodes, in each case even if such Nodes are
 *  propagated with or for interoperation with KNIME. The owner of a Node
 *  may freely choose the license terms applicable to such Node, including
 *  when such Node is propagated with or for interoperation with KNIME.
 * ------------------------------------------------------------------------
 *
 * History
 *   16 Dec 2010 (hornm): created
 */
package net.imglib2.algorithm.convolution;

import org.apache.commons.math3.linear.BlockRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.linear.SingularValueDecomposition;

import net.imglib2.Cursor;
import net.imglib2.IterableInterval;
import net.imglib2.RandomAccess;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.converter.Converter;
import net.imglib2.converter.read.ConvertedRandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.img.ImgFactory;
import net.imglib2.img.array.ArrayImgFactory;
import net.imglib2.ops.image.UnaryConstantRightAssignment;
import net.imglib2.ops.iterable.Sum;
import net.imglib2.ops.operation.binary.real.RealDivide;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.DoubleType;
import net.imglib2.view.Views;

/**
 * A collection of static methods in support of two-dimensional filters. TODO:
 * Make Operations?!
 * 
 * 
 * @author Roy Liu, hornm
 */
public final class FilterTools
{

	private FilterTools()
	{
		//
	}

	/**
	 * Creates a point support matrix. The top row consists of <tt>x</tt>
	 * coordinates, and the bottom row consists of <tt>y</tt> coordinates. The
	 * points range in a square where the origin is at the center.
	 * 
	 * @param supportRadius
	 *            the support radius.
	 * @return the point support matrix.
	 */
	final public static Img< DoubleType > createPointSupport( int supportRadius )
	{

		int support = supportRadius * 2 + 1;

		Img< DoubleType > res = new ArrayImgFactory< DoubleType >().create( new long[] { 2, support * support }, new DoubleType() );

		Cursor< DoubleType > cur = res.localizingCursor();

		while ( cur.hasNext() )
		{
			cur.fwd();
			if ( cur.getLongPosition( 0 ) == 0 )
			{
				cur.get().set( ( cur.getLongPosition( 1 ) / support ) - supportRadius );
			}
			else
			{
				cur.get().set( ( cur.getLongPosition( 1 ) % support ) - supportRadius );
			}
		}
		return res;
	}

	/**
	 * Creates the <tt>2&#215;2</tt> rotation matrix <br />
	 * <tt>-cos(theta) -sin(theta)</tt> <br />
	 * <tt>-sin(theta) cos(theta)</tt>. <br />
	 * 
	 * @param theta
	 *            the angle of rotation.
	 * @return the rotation matrix.
	 */
	final public static Img< DoubleType > createRotationMatrix( double theta )
	{

		Img< DoubleType > res = new ArrayImgFactory< DoubleType >().create( new long[] { 2, 2 }, new DoubleType() );

		RandomAccess2D< DoubleType > ra = new RandomAccess2D< DoubleType >( res );

		ra.get( 0, 0 ).set( ( float ) -Math.cos( theta ) );
		ra.get( 0, 1 ).set( ( float ) -Math.sin( theta ) );
		ra.get( 1, 0 ).set( ( float ) -Math.sin( theta ) );
		ra.get( 1, 1 ).set( ( float ) Math.cos( theta ) );

		return res;
	}

	final public static < T extends RealType< T > & NativeType< T >> Img< T > reshapeMatrix( long stride, Img< T > vector )
	{

		long yDim = vector.dimension( 0 ) / stride;

		Img< T > res = new ArrayImgFactory< T >().create( new long[] { stride, yDim }, vector.firstElement().createVariable() );

		Cursor< T > vecCur = vector.localizingCursor();
		RandomAccess< T > resRA = res.randomAccess();

		while ( vecCur.hasNext() )
		{
			vecCur.fwd();
			resRA.setPosition( vecCur.getLongPosition( 0 ) % stride, 1 );
			resRA.setPosition( vecCur.getLongPosition( 0 ) / stride, 0 );
			resRA.get().set( vecCur.get() );
		}
		return res;

	}

	public static < T extends RealType< T > & NativeType< T >> Img< T > getVector( Img< T > src, int[] pos, int vectorDim )
	{

		Img< T > vector = new ArrayImgFactory< T >().create( new long[] { src.dimension( vectorDim ) }, src.firstElement().createVariable() );

		Cursor< T > vecCur = vector.localizingCursor();
		RandomAccess< T > srcRA = src.randomAccess();

		srcRA.setPosition( pos );
		while ( vecCur.hasNext() )
		{
			vecCur.fwd();
			srcRA.setPosition( vecCur.getLongPosition( 0 ), vectorDim );
			vecCur.get().set( srcRA.get() );
		}
		return vector;

	}

	private static class RandomAccess2D< T extends RealType< T >>
	{

		private final RandomAccess< T > m_ra;

		public RandomAccess2D( Img< T > img )
		{
			m_ra = img.randomAccess();
		}

		public T get( int row, int col )
		{
			m_ra.setPosition( col, 1 );
			m_ra.setPosition( row, 0 );
			return m_ra.get();
		}

	}

	public static < T extends RealType< T >> void print2DMatrix( Img< T > img )
	{
		if ( img.numDimensions() < 2 ) { return; }
		RandomAccess< T > ra = img.randomAccess();
		for ( int x = 0; x < img.dimension( 0 ); x++ )
		{
			System.out.println( "" );
			ra.setPosition( x, 0 );
			for ( int y = 0; y < img.dimension( 1 ); y++ )
			{
				ra.setPosition( y, 1 );
				System.out.printf( " %+.4f", ra.get().getRealDouble() );
			}
		}
	}

	/**
	 * Creates a numDims dimensions image where all dimensions d<numDims are of
	 * size 1 and the last dimensions contains the vector.
	 * 
	 * @param <R>
	 * @param ar
	 * @param type
	 * @param numDims
	 *            number of dimensions
	 * @return
	 */
	public static < R extends RealType< R >> Img< R > vectorToImage( final RealVector ar, final R type, int numDims, ImgFactory< R > fac )
	{
		long[] dims = new long[ numDims ];

		for ( int i = 0; i < dims.length - 1; i++ )
		{
			dims[ i ] = 1;
		}
		dims[ dims.length - 1 ] = ar.getDimension();
		Img< R > res = fac.create( dims, type );
		Cursor< R > c = res.cursor();
		while ( c.hasNext() )
		{
			c.fwd();
			c.get().setReal( ar.getEntry( c.getIntPosition( numDims - 1 ) ) );
		}

		return res;
	}

	/**
	 * 
	 * @param <R>
	 * @param img
	 *            A two dimensional image.
	 * @return
	 */
	public static < R extends RealType< R >> RealMatrix toMatrix( final Img< R > img )
	{

		assert img.numDimensions() == 2;
		RealMatrix ret = new BlockRealMatrix( ( int ) img.dimension( 0 ), ( int ) img.dimension( 1 ) );
		Cursor< R > c = img.cursor();
		while ( c.hasNext() )
		{
			c.fwd();
			ret.setEntry( c.getIntPosition( 0 ), c.getIntPosition( 1 ), c.get().getRealDouble() );
		}
		return ret;
	}

	private static < K extends RealType< K >, KERNEL extends RandomAccessibleInterval< K >> SingularValueDecomposition isDecomposable( KERNEL kernel )
	{

		if ( kernel.numDimensions() != 2 )
			return null;

		final RealMatrix mKernel = new ImgBasedRealMatrix< K, KERNEL >( kernel );

		final SingularValueDecomposition svd = new SingularValueDecomposition( mKernel );

		if ( svd.getRank() > 1 )
			return null;

		return svd;

	}

	public static synchronized final < T extends RealType< T >, TT extends RandomAccessibleInterval< T >> void normalizeKernelInPlace( final TT kernel )
	{

		IterableInterval< T > iterable = Views.iterable( kernel );

		DoubleType dsum = new Sum< T, DoubleType >().compute( iterable.cursor(), new DoubleType() );

		if ( dsum.getRealDouble() == 0 )
			return;

		new UnaryConstantRightAssignment< T, DoubleType, T >( new RealDivide< T, DoubleType, T >() ).compute( iterable, dsum, iterable );
	}

	public static < K extends RealType< K >, KERNEL extends RandomAccessibleInterval< K >> RandomAccessibleInterval< DoubleType >[] decomposeKernel( KERNEL kernel )
	{

		SingularValueDecomposition svd = isDecomposable( kernel );
		if ( svd != null )
		{
			Img< DoubleType > vkernel;
			Img< DoubleType > ukernel;

			final RealVector v = svd.getV().getColumnVector( 0 );
			final RealVector u = svd.getU().getColumnVector( 0 );
			final double s = -Math.sqrt( svd.getS().getEntry( 0, 0 ) );
			v.mapMultiplyToSelf( s );
			u.mapMultiplyToSelf( s );
			vkernel = null;
			ukernel = null;

			// V -> horizontal
			vkernel = vectorToImage( v, new DoubleType(), 1, new ArrayImgFactory< DoubleType >() );
			// U -> vertical
			ukernel = vectorToImage( u, new DoubleType(), 2, new ArrayImgFactory< DoubleType >() );

			return new RandomAccessibleInterval[] { vkernel, ukernel };

		}
		else
		{

			ConvertedRandomAccessibleInterval< K, DoubleType > conv = new ConvertedRandomAccessibleInterval< K, DoubleType >( kernel, new Converter< K, DoubleType >()
			{

				@Override
				public void convert( K input, DoubleType output )
				{
					output.set( input.getRealDouble() );
				}
			}, new DoubleType() );

			return new RandomAccessibleInterval[] { conv };
		}

	}

}
