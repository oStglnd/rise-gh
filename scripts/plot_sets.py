
import os
import json

plot_unit   = '6 hours'
plot_suffix = '6h'

# define plot categories and params
plot_cats = [
    # EXAMPLE: ('title', 'type', 'filter (RegEx)', 'units', 'palette', 'alpha', 'filename'),
    {
        'title':'DC, TA01-GP101 - Flow, {}'.format(plot_unit),
        'type':'flow',
        'filter':'TA01_GP101|FF01_GP101',
        'units':'l/s', 
        'palette':'magma',
        'alpha':1.0,
        'fname':'dc_flow_{}.png'.format(plot_suffix)
    },
    {
        'title':'DC, TA01 - Flow, {}'.format(plot_unit),
        'type':'state',
        'filter':'TA01_output',
        'units':'...', 
        'palette':'magma',
        'alpha':1.0,
        'fname':'dc_outp_{}.png'.format(plot_suffix)
    },
    {
        'title':'GH, SP - Flow, {}'.format(plot_unit),
        'type':'flow',
        'filter':'TA01',
        'units':'l/s', 
        'palette':'YlOrRd_r',
        'alpha':0.9,
        'fname':'gh_ta01_flow_{}.png'.format(plot_suffix)
    },
    {
        'title':'DC, SP - Flow, {}'.format(plot_unit),
        'type':'flow',
        'filter':'DC_SP',
        'units':'l/s', 
        'palette':'YlOrRd_r',
        'alpha':0.8,
        'fname':'dc_sp_flow_{}.png'.format(plot_suffix)
    },
    {
        'title':'DC - Temperatures, {}'.format(plot_unit),
        'type':'temperatures',
        'filter':'GM401',
        'units':'$^\circ$C', 
        'palette':'YlOrRd_r',
        'alpha':0.9,
        'fname':'dc_temp_{}.png'.format(plot_suffix)
    },
    {
        'title':'DC - Humidity, {}'.format(plot_unit),
        'type':'humidity',
        'filter':'GM401',
        'units':'$\%$', 
        'palette':'crest',
        'alpha':0.9,
        'fname':'dc_humid_{}.png'.format(plot_suffix)
    },
    {
        'title':'GH - Avg., {}'.format(plot_unit),
        'type':'all',
        'filter':'X',
        'units':'$^\circ$C, $\%$', 
        'palette':'mako',
        'alpha':1.0,
        'fname':'ghavg_all_{}.png'.format(plot_suffix)
    },
    {
        'title':'DC - , {}'.format(plot_unit),
        'type':'all',
        'filter':'DC_GT401_GM401',
        'units':'$^\circ$C, $\%$', 
        'palette':'mako',
        'alpha':1.0,
        'fname':'dc_all_{}.png'.format(plot_suffix)
    },
    {
        'title':'GH & OUT - Temperatures, {}'.format(plot_unit),
        'type':'temperatures',
        'filter':'X|GT301|SMHI',
        'units':'$^\circ$C', 
        'palette':'YlOrRd_r',
        'alpha':1.0,
        'fname':'ghout_temp_{}.png'.format(plot_suffix)
    },
    {
        'title':'GH, DC & OUT - Humidity, {}'.format(plot_unit),
        'type':'humidity',
        'filter':'X|DC_GT401_GM401|SMHI',
        'units':'$\%$', 
        'palette':'crest',
        'alpha':1.0,
        'fname':'ghdcout_hum_{}.png'.format(plot_suffix)
    },
    {
        'title':'OUT - Temperature, {}'.format(plot_unit),
        'type':'temperatures',
        'filter':'GT301|SMHI',
        'units':'$^\circ$C', 
        'palette':'mako',
        'alpha':1.0,
        'fname':'out_temp_{}.png'.format(plot_suffix)
    },
    {
        'title':'GH - Temperature, {}'.format(plot_unit),
        'type':'temperatures',
        'filter':'X',
        'units':'$^\circ$C', 
        'palette':'mako',
        'alpha':1.0,
        'fname':'ghavg_temp_{}.png'.format(plot_suffix)
    }
]

home_path = os.path.dirname(os.getcwd())
json_path = home_path + '\\misc\\plots.json'

json_obj = json.dumps(plot_cats)
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(json_obj, f, ensure_ascii=False, indent=4)