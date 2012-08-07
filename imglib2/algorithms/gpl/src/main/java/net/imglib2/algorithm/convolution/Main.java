package net.imglib2.algorithm.convolution;

import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgFactory;
import net.imglib2.outofbounds.OutOfBoundsMirrorFactory;
import net.imglib2.outofbounds.OutOfBoundsMirrorFactory.Boundary;
import net.imglib2.type.numeric.real.DoubleType;

public class Main
{
	public static void main( String[] args )
	{

		Img< DoubleType > img = new ArrayImgFactory< DoubleType >().create( new long[] { 1025, 1000, 4 }, new DoubleType() );

		Img[] kernels = new Img[] { ConstantFilter.Laplacian.createImage( 0 ), ConstantFilter.Sobel.createImage( 0 ) };

		ApplyMultiKernelConvolutionOp< DoubleType, DoubleType, DoubleType, Img< DoubleType >> multiKernelOp = new ApplyMultiKernelConvolutionOp< DoubleType, DoubleType, DoubleType, Img< DoubleType >>( new DirectImageConvolution< DoubleType, DoubleType, DoubleType, Img< DoubleType >, Img< DoubleType >>( true, new OutOfBoundsMirrorFactory< DoubleType, Img< DoubleType > >( Boundary.SINGLE ) ), new DirectImageConvolution< DoubleType, DoubleType, DoubleType, Img< DoubleType >, Img< DoubleType >>( true, new OutOfBoundsMirrorFactory< DoubleType, Img< DoubleType > >( Boundary.SINGLE ) ), Mode.ITERATE, kernels );

		Img< DoubleType > res = multiKernelOp.compute( img, img.factory().create( img, new DoubleType() ) );
	}
}
