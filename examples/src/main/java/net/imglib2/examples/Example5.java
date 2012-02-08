package net.imglib2.examples;

import ij.ImageJ;

import java.io.File;

import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgFactory;
import net.imglib2.img.display.imagej.ImageJFunctions;
import net.imglib2.io.ImgOpener;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.type.numeric.RealType;

import mpicbg.imglib.algorithm.CanvasImage;
import mpicbg.imglib.outofbounds.OutOfBoundsStrategyFactory;
import mpicbg.imglib.outofbounds.OutOfBoundsStrategyMirrorExpWindowingFactory;
import mpicbg.imglib.outofbounds.OutOfBoundsStrategyMirrorFactory;
import mpicbg.imglib.outofbounds.OutOfBoundsStrategyPeriodicFactory;
import mpicbg.imglib.outofbounds.OutOfBoundsStrategyValueFactory;
import mpicbg.imglib.util.Util;

/**
 * Illustrate what the outside strategies do
 *
 * @author Stephan Preibisch &amp; Stephan Saalfeld
 *
 */
public class Example5
{
	public Example5()
	{
		// define the file to open
		File file = new File( "DrosophilaWingSmall.tif" );

		// open with ImgOpener using an ArrayImgFactory
		Img< FloatType > image = new ImgOpener().openLOCIFloatType( file.getAbsolutePath(), new ArrayImgFactory< T >() );

		// test serveral out of bounds strategies
		testCanvas( image, new OutOfBoundsStrategyValueFactory< FloatType >() );
		testCanvas( image, new OutOfBoundsStrategyValueFactory< FloatType >( new FloatType( 128 ) ) );
		testCanvas( image, new OutOfBoundsStrategyMirrorFactory< FloatType >() );
		testCanvas( image, new OutOfBoundsStrategyPeriodicFactory< FloatType >() );
		testCanvas( image, new OutOfBoundsStrategyMirrorExpWindowingFactory< FloatType >( 0.5f ) );
	}

	public < T extends RealType< T >> void testCanvas( final Img< T > img, final OutOfBoundsStrategyFactory< T > outofboundsFactory )
	{
		final int[] newSize = new int[img.getNumDimensions()];

		for ( int d = 0; d < img.getNumDimensions(); ++d )
			newSize[d] = Util.round( img.getDimension( d ) * 3 );

		final CanvasImage< T > canvas = new CanvasImage< T >( img, newSize, outofboundsFactory );

		if ( canvas.checkInput() && canvas.process() )
		{
			Img< T > out = canvas.getResult();

			final String name = outofboundsFactory.getClass().getSimpleName() + " took " + canvas.getProcessingTime() + " ms.";
			ImageJFunctions.show( out ).setTitle( name );
		} else
		{
			System.out.println( canvas.getErrorMessage() );
		}
	}

	public static void main( String[] args )
	{
		// open an ImageJ window
		new ImageJ();

		// run the example
		new Example5();
	}
}
