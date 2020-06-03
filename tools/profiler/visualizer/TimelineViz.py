import numpy as np
from bokeh.models import ColumnDataSource, DataRange1d, Plot, LinearAxis, Grid, Legend, Text
from bokeh.plotting import figure
from bokeh.models.glyphs import Quad
from bokeh.io import curdoc, show, output_file
from bokeh.models.ranges import FactorRange
from bokeh.palettes import Category20

from visualizer.Config import get_config

def create(listEvntsName, evntStartTime, evntEndTime, listTagsName, title, plot_height=get_config('timeline_plot_height'), plot_width = get_config('timeline_plot_width'), tools=None, colorPallete=None, x_axis_label = 'Time', y_axis_label = 'Calls', border_color=None, fim_plot=None):
    ''' Creates Timeline Visualization '''

    numEvnts = len(listEvntsName)

    if colorPallete == None:
        colorPallete = Category20[20]
    assert numEvnts == len(evntStartTime), 'Num of events do not match'
    assert numEvnts == len(evntEndTime), 'Num of events do not match'

    xdr = DataRange1d()
    if(fim_plot):
        ydr = DataRange1d(end = numEvnts+3)
    else:
        ydr = DataRange1d()

    plot = figure(title=title, x_range=xdr, y_range=ydr, plot_width=plot_width, plot_height=plot_height,min_border=0)
    plot.xaxis.axis_label = x_axis_label
    plot.yaxis.axis_label = y_axis_label
    plot.yaxis.ticker = [ (i+0.5) for i in range(len(listEvntsName))]
    plot.yaxis.major_label_overrides = {(i+0.5):ticker_name for i,ticker_name in enumerate(listEvntsName) }
    plot.border_fill_color = border_color
    glyphs = []
    clr_ctr = 0
    legend_items = []
    for i in range(numEvnts):
        for j in range(len(listTagsName[i])):
            source = ColumnDataSource(dict(
                left=evntStartTime[i][j],
                right=evntEndTime[i][j],
                top = [i+1] * len(evntStartTime[i][j]),
                bottom = [i] * len(evntStartTime[i][j]),
				names = [listTagsName[i][j]] * len(evntStartTime[i][j])
            ))
            glyph = plot.quad(left="left", right="right", top="top", bottom="bottom",fill_color=colorPallete[clr_ctr], line_color=colorPallete[clr_ctr],
                              source=source)
            if(fim_plot):
                if(listTagsName[i][j].endswith('.cpp')):
                    glyph.visible=False
                else:
                    labels = Text(x='left', y='top', text='names', x_offset=0, y_offset=5, angle = 0.6, text_font_size='10pt')
                    plot.add_glyph(source, labels)

            clr_ctr+=1
            legend_items.append((listTagsName[i][j],[glyph]))

    legend = Legend(items=legend_items, location=(30, 0),click_policy="hide")

    plot.add_layout(legend, 'right')
    xaxis = LinearAxis() 
    plot.add_layout(Grid(dimension=0, ticker=xaxis.ticker))

    return plot


if __name__=='__main__':
    output_file('testTimelineViz.html', title = 'TimelineViz', mode = 'inline')

    #Generate events
    listEvntsName = ['EvntType1', 'EvntType2', 'EvnType3']
    x = np.linspace(1,9,9,dtype='int')
    evntStartTime = np.array([x**2, x**2 + 2, x**2 + 5])
    evntEndTime = evntStartTime + np.random.randint(1,5,(3,9))
    listTagsName = [['EvntType1'], ['EvntType2'], ['EvnType3']]

    plot = create(listEvntsName, evntStartTime, evntEndTime, listTagsName, 'TimelineViz', plot_height=200 )
    show(plot)
