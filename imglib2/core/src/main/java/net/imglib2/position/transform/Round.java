/**
 * Copyright (c) 2009--2012, ImgLib2 developers
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 * Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.  Redistributions in binary
 * form must reproduce the above copyright notice, this list of conditions and
 * the following disclaimer in the documentation and/or other materials
 * provided with the distribution.  Neither the name of the imglib project nor
 * the names of its contributors may be used to endorse or promote products
 * derived from this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */
package net.imglib2.position.transform;

import net.imglib2.Localizable;
import net.imglib2.Positionable;
import net.imglib2.RealLocalizable;
import net.imglib2.RealPositionable;

/**
 * A {@link RealPositionable} that drives a {@link Positionable} to its
 * round discrete coordinates:
 * 
 * f = r < 0 ? (long)( r - 0.5 ) : (long)( r + 0.5 )
 * 
 * @author Stephan Saalfeld <saalfeld@mpi-cbg.de>
 */
public class Round< LocalizablePositionable extends Localizable & Positionable > extends AbstractPositionableTransform< LocalizablePositionable >
{
	public Round( final LocalizablePositionable target )
	{
		super( target );
	}
	
	public Round( final RealLocalizable origin, final LocalizablePositionable target )
	{
		super( target );
		
		origin.localize( position );
		for ( int d = 0; d < n; ++d )
			discrete[ d ] = round( position[ d ] );
		target.setPosition( discrete );
	}
	
	final static protected long round( final double r )
	{
		return r < 0 ? ( long )( r - 0.5 ) : ( long )( r + 0.5 );
	}
	
	final static protected long round( final float r )
	{
		return r < 0 ? ( long )( r - 0.5f ) : ( long )( r + 0.5f );
	}
	
	final static protected void round( final double[] r, final long[] f )
	{
		for ( int d = 0; d < r.length; ++d )
			f[ d ] = round( r[ d ] );
	}
	
	final static protected void round( final float[] r, final long[] f )
	{
		for ( int d = 0; d < r.length; ++d )
			f[ d ] = round( r[ d ] );
	}
	
	final static protected void round( final RealLocalizable r, final long[] f )
	{
		for ( int d = 0; d < f.length; ++d )
			f[ d ] = round( r.getDoublePosition( d ) );
	}
	
	
	/* RealPositionable */
	
	@Override
	public void move( final float distance, final int d )
	{
		final double realPosition = position[ d ] + distance;
		final long roundPosition = round( realPosition );
		position[ d ] = realPosition;
		final long roundDistance = roundPosition - target.getIntPosition( d );
		if ( roundDistance == 0 )
			return;
		else
			target.move( roundDistance, d );
	}

	@Override
	public void move( final double distance, final int d )
	{
		final double realPosition = position[ d ] + distance;
		final long roundPosition = round( realPosition );
		position[ d ] = realPosition;
		final long roundDistance = roundPosition - target.getIntPosition( d );
		if ( roundDistance == 0 )
			return;
		else
			target.move( roundDistance, d );
	}

	@Override
	public void move( final RealLocalizable localizable )
	{
		for ( int d = 0; d < n; ++d )
		{
			final double realPosition = position[ d ] + localizable.getDoublePosition( d );
			final long floorPosition = round( realPosition );
			position[ d ] = realPosition;
			discrete[ d ] = floorPosition - target.getLongPosition( d );
		}
		target.move( discrete );
	}

	@Override
	public void move( final float[] distance )
	{
		for ( int d = 0; d < n; ++d )
		{
			final double realPosition = position[ d ] + distance[ d ];
			final long floorPosition = round( realPosition );
			position[ d ] = realPosition;
			discrete[ d ] = floorPosition - target.getLongPosition( d );
		}
		target.move( discrete );
	}

	@Override
	public void move( final double[] distance )
	{
		for ( int d = 0; d < n; ++d )
		{
			final double realPosition = position[ d ] + distance[ d ];
			final long floorPosition = round( realPosition );
			position[ d ] = realPosition;
			discrete[ d ] = floorPosition - target.getLongPosition( d );
		}
		target.move( discrete );
	}

	@Override
	public void setPosition( final RealLocalizable localizable )
	{
		for ( int d = 0; d < n; ++d )
		{
			final double realPosition = localizable.getDoublePosition( d );
			position[ d ] = realPosition;
			discrete[ d ] = round( realPosition );
		}
		target.setPosition( discrete );
	}

	@Override
	public void setPosition( final float[] pos )
	{
		for ( int d = 0; d < n; ++d )
		{
			final float realPosition = pos[ d ];
			position[ d ] = realPosition;
			discrete[ d ] = round( realPosition );
		}
		target.setPosition( discrete );
	}

	@Override
	public void setPosition( final double[] pos )
	{
		for ( int d = 0; d < n; ++d )
		{
			final double realPosition = pos[ d ];
			position[ d ] = realPosition;
			discrete[ d ] = round( realPosition );
		}
		target.setPosition( discrete );
	}

	@Override
	public void setPosition( final float position, final int d )
	{
		this.position[ d ] = position;
		target.setPosition( round( position ), d );
	}

	@Override
	public void setPosition( final double position, final int d )
	{
		this.position[ d ] = position;
		target.setPosition( round( position ), d );
	}
}
