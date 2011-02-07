package mpicbg.imglib.ui;

import java.awt.Adjustable;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Image;
import java.awt.Toolkit;
import java.awt.Window;
import java.awt.event.AdjustmentEvent;
import java.awt.event.AdjustmentListener;
import java.util.ArrayList;
import java.util.List;

import javax.swing.BoxLayout;
import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.JScrollBar;
import javax.swing.border.TitledBorder;

import mpicbg.imglib.container.Img;
import mpicbg.imglib.container.ImgFactory;
import mpicbg.imglib.container.array.ArrayContainerFactory;
import mpicbg.imglib.display.ARGBScreenImage;
import mpicbg.imglib.display.RealARGBConverter;
import mpicbg.imglib.display.XYProjector;
import mpicbg.imglib.io.LOCI;
import mpicbg.imglib.type.numeric.ARGBType;
import mpicbg.imglib.type.numeric.real.FloatType;

public class ImgPanel extends JPanel {

	public class ImgData {
		public String name;
		public Img<FloatType> img;
		public ImgPanel owner;
		public int width, height;
		public ARGBScreenImage screenImage;
		public RealARGBConverter converter;
		public XYProjector<FloatType, ARGBType> projector;

		public ImgData(final String name, final Img<FloatType> img,
			final ImgPanel owner)
		{
			this.name = name;
			this.img = img;
			this.owner = owner;
			width = (int) img.dimension(0);
			height = (int) img.dimension(1);
			screenImage = new ARGBScreenImage(width, height);
			final int min = 0, max = 255;
			converter = new RealARGBConverter(min, max);
			projector = new XYProjector<FloatType, ARGBType>(img,
				screenImage, converter);
			projector.map();
		}
	}

	public class SliderPanel extends JPanel {
		public SliderPanel(final ImgData imgData) {
			setBorder(new TitledBorder(imgData.name));
			setLayout(new BoxLayout(this, BoxLayout.Y_AXIS));
			// add one slider per dimension beyond the first two
			for (int d=2; d<imgData.img.numDimensions(); d++) {
				final int dimLength = (int) imgData.img.dimension(d);
				final JScrollBar bar = new JScrollBar(Adjustable.HORIZONTAL,
					0, 1, 0, dimLength);
				final int dim = d;
				bar.addAdjustmentListener(new AdjustmentListener() {
					@Override
					public void adjustmentValueChanged(AdjustmentEvent e) {
						final int value = bar.getValue();
						imgData.projector.setPosition(value, dim);
						imgData.projector.map();
						System.out.println("dim #" + dim + ": value->" + value);//TEMP
						imgData.owner.repaint();
					}
				});
				add(bar);
			}
		}
	}

	private List<ImgData> images = new ArrayList<ImgData>();
	private int maxWidth = 0, maxHeight = 0;

	public ImgPanel() {
		setLayout(new BoxLayout(this, BoxLayout.Y_AXIS));
		add(new JPanel() { // image canvas
			@Override
			public void paint(Graphics g) {
				for (final ImgData imgData : images) {
					final Image image = imgData.screenImage.image();
					g.drawImage(image, 0, 0, this);
				}
			}

			@Override
			public Dimension getPreferredSize() {
				return new Dimension(maxWidth, maxHeight);
			}
		});
	}

	public void addImage(final String name, final Img<FloatType> img) {
		final ImgData imgData = new ImgData(name, img, this);
		images.add(imgData);
		if (imgData.width > maxWidth) maxWidth = imgData.width;
		if (imgData.height > maxHeight) maxHeight = imgData.height;
		add(new SliderPanel(imgData));
	}

	public static final void main(final String[] args) {
		final String[] paths = {
			"/Users/curtis/data/mitosis-test.ipw",
			"/Users/curtis/data/z-series.ome.tif"
		};
		final JFrame frame = new JFrame("ImgPanel Test Frame");
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		final ImgPanel imgPanel = new ImgPanel();
		for (String path : paths) {
			imgPanel.addImage(path, loadImage(path));
		}
		frame.setContentPane(imgPanel);
		frame.pack();
		center(frame);
		frame.setVisible(true);
	}

	private static Img<FloatType> loadImage(String path) {
		final ImgFactory<FloatType> imgFactory =
			new ArrayContainerFactory<FloatType>();
		return LOCI.openLOCIFloatType(path, imgFactory);
	}

	private static void center(final Window win) {
		final Dimension size = win.getSize();
		final Dimension screen = Toolkit.getDefaultToolkit().getScreenSize();
		final int w = (screen.width - size.width) / 2;
		final int h = (screen.height - size.height) / 2;
		win.setLocation(w, h);
	}

}