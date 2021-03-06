/*
 * #%L
 * ImgLib: a general-purpose, multidimensional image processing library.
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

package mpicbg.imglib.cursor.dynamic;

import mpicbg.imglib.container.basictypecontainer.DataAccess;
import mpicbg.imglib.container.dynamic.DynamicContainer;
import mpicbg.imglib.container.dynamic.DynamicContainerAccessor;
import mpicbg.imglib.cursor.Cursor;
import mpicbg.imglib.cursor.CursorImpl;
import mpicbg.imglib.image.Image;
import mpicbg.imglib.type.Type;

/**
 * TODO
 *
 * @author Stephan Preibisch
 */
public class DynamicCursor<T extends Type<T>> extends CursorImpl<T> implements Cursor<T>
{
	protected final T type;
	protected final DynamicContainer<T,? extends DataAccess> container;
	protected final DynamicContainerAccessor accessor;
	int internalIndex;
	
	public DynamicCursor( final DynamicContainer<T,? extends DynamicContainerAccessor> container, final Image<T> image, final T type ) 
	{
		super( container, image );
		
		this.type = type;
		this.container = container;
	
		accessor = container.createAccessor();
		
		reset();
	}
	
	public DynamicContainerAccessor getAccessor() { return accessor; }

	@Override
	public T getType() { return type; }

	@Override
	public boolean hasNext() { return internalIndex < container.getNumPixels() - 1; }

	@Override
	public void fwd( final long steps ) 
	{ 
		internalIndex += steps;
		accessor.updateIndex( internalIndex );
	}

	@Override
	public void fwd() 
	{ 
		++internalIndex; 
		accessor.updateIndex( internalIndex );
	}

	@Override
	public void close() 
	{ 
		isClosed = true;
		internalIndex = Integer.MAX_VALUE;
	}

	@Override
	public void reset()
	{		
		type.updateIndex( 0 );
		internalIndex = 0;
		type.updateContainer( this );
		accessor.updateIndex( internalIndex );
		internalIndex = -1;
		isClosed = false;
	}
	
	public int getInternalIndex() { return internalIndex; }

	@Override
	public DynamicContainer<T,?> getStorageContainer(){ return container; }

	@Override
	public int getStorageIndex() { return internalIndex; }
	
	@Override
	public String toString() { return type.toString(); }		
}
