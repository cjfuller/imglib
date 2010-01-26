/**
 * Copyright (c) 2009--2010, Stephan Preibisch & Stephan Saalfeld
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 * Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.  Redistributions in binary
 * form must reproduce the above copyright notice, this list of conditions and
 * the following disclaimer in the documentation and/or other materials
 * provided with the distribution.  Neither the name of the Fiji project nor
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
 *
 * @author Stephan Preibisch & Stephan Saalfeld
 */
package mpicbg.imglib.cursor;

import java.util.Iterator;

import mpicbg.imglib.container.Container;
import mpicbg.imglib.container.array.Array;
import mpicbg.imglib.container.cube.Cube;
import mpicbg.imglib.cursor.vector.Dimensionality;
import mpicbg.imglib.image.Image;
import mpicbg.imglib.type.Type;

/**
 * The {@link Cursor} is responsible for iterating over the image. Therefore it has to be implemented 
 * for each type of {@link Container} like {@link Array}, {@link Cube}, ... 
 * 
 * The {@link Cursor} does not know which {@link Type} of {@link Image} it is working on as there this
 * is not important for its positioning. It is typed to the {@link Type} so that the {@link Cursor} is
 * able to return the correct instance of {@link Type} when calling the getType() method.
 * 
 * The {@link Cursor} class itself is only capable of iterating over all pixels of the {@link Image}, 
 * there is no guaranteed order in which the pixels are iterated, this depends on the implementation for
 * the specific {@link Container}.
 *  
 * @author Stephan Preibisch & Stephan Saalfeld
 *
 * @param <T> - the {@link Type} this {@link Cursor} works on
 */
public interface Cursor<T extends Type<T>> extends Iterator<T>, java.lang.Iterable<T>, Iterable, Dimensionality
{	
	public void reset();			
	public boolean isActive();	

	public Image<T> getImage();
	public T getType();
	public int getArrayIndex();
	public int getStorageIndex();
	public Container<T> getStorageContainer();
	public void setDebug( final boolean debug );
	public int[] createPositionArray();
	
	public void close();
}