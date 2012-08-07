package net.imglib2.algorithm.convolution;

import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.ops.UnaryOperation;
import net.imglib2.ops.image.AdditionalDimUnaryOperation;
import net.imglib2.type.numeric.RealType;

public class ApplyMultiKernelConvolutionOp< T extends RealType< T >, O extends RealType< O >, K extends RealType< K >, KERNEL extends RandomAccessibleInterval< K >> implements UnaryOperation< Img< T >, Img< O >>
{

	private final Mode m_mode;

	private KERNEL[] m_kernels;

	private Convolution< T, O, K, Img< T >, Img< O >, KERNEL >[] m_ops;

	private Convolution< O, O, K, Img< O >, Img< O >, KERNEL > m_followConv;

	private Img< O > m_buffer;

	/**
	 * First init is computed then n-times the follower. We need to distinguish
	 * because of generic implementations (in maybe != out)
	 * 
	 * @param init
	 * @param follower
	 * @param mode
	 * @param kernels
	 */
	@SuppressWarnings( "unchecked" )
	public ApplyMultiKernelConvolutionOp( Convolution< T, O, K, Img< T >, Img< O >, KERNEL > init, Convolution< O, O, K, Img< O >, Img< O >, KERNEL > follower, Mode mode, KERNEL[] kernels )
	{
		this.m_mode = mode;
		this.m_kernels = kernels;
		if ( m_mode == Mode.ADD_DIM )
		{
			m_ops = new Convolution[ m_kernels.length ];
			for ( int k = 0; k < m_ops.length; k++ )
			{
				m_ops[ k ] = ( Convolution< T, O, K, Img< T >, Img< O >, KERNEL > ) init.copy();
				m_ops[ k ].setKernel( m_kernels[ k ] );
			}

		}
		else
		{
			m_ops = new Convolution[ 1 ];
			m_ops[ 0 ] = init;
			m_ops[ 0 ].setKernel( m_kernels[ 0 ] );
		}

		m_followConv = follower;
	}

	@Override
	public Img< O > compute( Img< T > input, Img< O > output )
	{

		if ( m_kernels.length == 1 )
		{
			return m_ops[ 0 ].compute( input, output );
		}
		else if ( m_mode == Mode.ITERATE )
		{

			initBuffer( input, output );
			Img< O > tmpOutput = output;

			Img< O > tmpInput = m_buffer;
			Img< O > tmp;

			if ( m_kernels.length % 2 == 0 )
			{
				tmpOutput = m_buffer;
				tmpInput = output;
			}

			m_ops[ 0 ].compute( input, tmpOutput );

			for ( int i = 1; i < m_kernels.length; i++ )
			{
				tmp = tmpInput;
				tmpInput = tmpOutput;
				tmpOutput = tmp;
				m_followConv.setKernel( m_kernels[ i ] );
				m_followConv.compute( tmpInput, tmpOutput );
			}

			return output;
		}
		else if ( m_mode == Mode.ADD_DIM )
		{

			AdditionalDimUnaryOperation< T, O, Img< T >> add = new AdditionalDimUnaryOperation< T, O, Img< T >>( output.firstElement().createVariable(), output.factory(), m_ops );

			add.compute( input, output );

			return output;
		}
		else
		{
			throw new IllegalArgumentException( "Filtering mode unknown in ConvolverNodeModel" );
		}
	}

	private void initBuffer( Img< T > input, Img< O > output )
	{
		if ( m_buffer == null || !equalsInterval( output, m_buffer ) )
		{
			m_buffer = output.factory().create( output, output.firstElement() );
		}

	}

	private boolean equalsInterval( Img< O > i1, Img< O > i2 )
	{

		for ( int d = 0; d < i1.numDimensions(); d++ )
			if ( i1.dimension( d ) != i2.dimension( d ) )
				return false;
		return true;
	}

	@SuppressWarnings( "unchecked" )
	@Override
	public UnaryOperation< Img< T >, Img< O >> copy()
	{
		return new ApplyMultiKernelConvolutionOp< T, O, K, KERNEL >( ( Convolution< T, O, K, Img< T >, Img< O >, KERNEL > ) m_ops[ 0 ].copy(), ( Convolution< O, O, K, Img< O >, Img< O >, KERNEL > ) m_followConv.copy(), m_mode, m_kernels );
	}

}
