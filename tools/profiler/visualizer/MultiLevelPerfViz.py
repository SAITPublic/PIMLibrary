from bokeh.transform import factor_cmap
from bokeh.models import FactorRange, ColumnDataSource
from bokeh.io import show, output_file 
from bokeh.plotting import figure
from bokeh.palettes import viridis

import numpy as np


def create(listLevel0, listLevel1, data, title, y_label, plot_height = 250, 
                                tools = None, colorPalletteLevel1 = None):
    ''' Creates a bokeh figure for multi-level perf data 
        listLevel0 = list of names of Higher level category (e.g. LayerName)
        listLevel1 = list of names of Lower level category (e.g. Kerneltime, Data Transfer Time)
        data = 2D numpy array for data arraged in (Highlevel,Lowlevel) format
        title = Title of the figure
        plot_height = Height for generated bokeh plot (Default = 250)
        tools = Set of tools to display along with plot (Defalt = All Bokeh tools)
        colorPalletteLevel1 = a list of colors to use for Lower level category (Default = sample from virdis)
    '''
    
    assert data.shape == (len(listLevel0), len(listLevel1)) 

    if colorPalletteLevel1 == None:
        colorPalletteLevel1 = viridis(len(listLevel1))
    
    # List of tuples for x axis (e.g. [(vgg,conf0),(vgg,conf1)])
    x = [(lvl0,lvl1) for lvl0 in listLevel0 for lvl1 in listLevel1]
    
    # ColumnData for bokeh
    source = ColumnDataSource(data=dict(x=x, counts=data.flatten()))
    
    p = figure(x_range=FactorRange(*x), plot_height=plot_height, title=title, tools=tools)
    
    p.vbar(x='x', top='counts', width=0.9, source=source, line_color="white",
           fill_color= factor_cmap('x', palette=colorPalletteLevel1, factors=listLevel1, start=1))
    
    p.y_range.start = 0
    p.x_range.range_padding = 0.1
    p.xaxis.major_label_orientation = 1
    p.xgrid.grid_line_color = None
    p.yaxis.axis_label = y_label
    p.toolbar.logo = None 
    
    return p

if __name__=='__main__':
    #Test create multiLevelPerfViz

    output_file(filename='testMultilevelPerfViz.html', title='MultiLevelPerfViz',mode='inline')

    networks = ['LSTM0','LSTM1', 'LSTM2']
    confs = ['conf1','conf2','conf3','conf4','conf5','conf6','conf7']
    
    #Radom data
    data = np.random.randint(low = 1, high = 1000, size=(len(networks), len(confs)))
    
    fig = create(networks, confs, data, "Perf. for multiple runs", "Time (ms)",
                                          tools= "wheel_zoom,box_zoom,save,reset")
    
    show(fig)
