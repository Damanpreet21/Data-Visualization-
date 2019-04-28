
#                        FIFA WORLD CUP 2018


```python
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

import plotly.offline as py
import plotly.graph_objs as go

```

Importing dataset and preparation of data

Load dataset


```python
# Importing the dataset 
dataset=pd.read_csv("https://raw.githubusercontent.com/fifa-players/Fifa/master/CompleteDataset.csv")
```

    /Users/ashinbasheer/anaconda3/lib/python3.7/site-packages/IPython/core/interactiveshell.py:3020: DtypeWarning:
    
    Columns (23,35) have mixed types. Specify dtype option on import or set low_memory=False.
    


Trying to get the head of the dataset


```python
dataset.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>Name</th>
      <th>Age</th>
      <th>Photo</th>
      <th>Nationality</th>
      <th>Flag</th>
      <th>Overall</th>
      <th>Potential</th>
      <th>Club</th>
      <th>Club Logo</th>
      <th>...</th>
      <th>RB</th>
      <th>RCB</th>
      <th>RCM</th>
      <th>RDM</th>
      <th>RF</th>
      <th>RM</th>
      <th>RS</th>
      <th>RW</th>
      <th>RWB</th>
      <th>ST</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Cristiano Ronaldo</td>
      <td>32</td>
      <td>https://cdn.sofifa.org/48/18/players/20801.png</td>
      <td>Portugal</td>
      <td>https://cdn.sofifa.org/flags/38.png</td>
      <td>94</td>
      <td>94</td>
      <td>Real Madrid CF</td>
      <td>https://cdn.sofifa.org/24/18/teams/243.png</td>
      <td>...</td>
      <td>61.0</td>
      <td>53.0</td>
      <td>82.0</td>
      <td>62.0</td>
      <td>91.0</td>
      <td>89.0</td>
      <td>92.0</td>
      <td>91.0</td>
      <td>66.0</td>
      <td>92.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>L. Messi</td>
      <td>30</td>
      <td>https://cdn.sofifa.org/48/18/players/158023.png</td>
      <td>Argentina</td>
      <td>https://cdn.sofifa.org/flags/52.png</td>
      <td>93</td>
      <td>93</td>
      <td>FC Barcelona</td>
      <td>https://cdn.sofifa.org/24/18/teams/241.png</td>
      <td>...</td>
      <td>57.0</td>
      <td>45.0</td>
      <td>84.0</td>
      <td>59.0</td>
      <td>92.0</td>
      <td>90.0</td>
      <td>88.0</td>
      <td>91.0</td>
      <td>62.0</td>
      <td>88.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>Neymar</td>
      <td>25</td>
      <td>https://cdn.sofifa.org/48/18/players/190871.png</td>
      <td>Brazil</td>
      <td>https://cdn.sofifa.org/flags/54.png</td>
      <td>92</td>
      <td>94</td>
      <td>Paris Saint-Germain</td>
      <td>https://cdn.sofifa.org/24/18/teams/73.png</td>
      <td>...</td>
      <td>59.0</td>
      <td>46.0</td>
      <td>79.0</td>
      <td>59.0</td>
      <td>88.0</td>
      <td>87.0</td>
      <td>84.0</td>
      <td>89.0</td>
      <td>64.0</td>
      <td>84.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>L. Suárez</td>
      <td>30</td>
      <td>https://cdn.sofifa.org/48/18/players/176580.png</td>
      <td>Uruguay</td>
      <td>https://cdn.sofifa.org/flags/60.png</td>
      <td>92</td>
      <td>92</td>
      <td>FC Barcelona</td>
      <td>https://cdn.sofifa.org/24/18/teams/241.png</td>
      <td>...</td>
      <td>64.0</td>
      <td>58.0</td>
      <td>80.0</td>
      <td>65.0</td>
      <td>88.0</td>
      <td>85.0</td>
      <td>88.0</td>
      <td>87.0</td>
      <td>68.0</td>
      <td>88.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>M. Neuer</td>
      <td>31</td>
      <td>https://cdn.sofifa.org/48/18/players/167495.png</td>
      <td>Germany</td>
      <td>https://cdn.sofifa.org/flags/21.png</td>
      <td>92</td>
      <td>92</td>
      <td>FC Bayern Munich</td>
      <td>https://cdn.sofifa.org/24/18/teams/21.png</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 75 columns</p>
</div>



The dataset ‘CompleteDataset.csv’ contains information about 17981 players in total and 75 attributes associated with
those players.

This dataset was obtained from the kaggle page here where they have scraped the data from this https://sofifa.com/

Majority of the attributes are self-explanatory like the Flag, Age, nationality etc. The attributes like Acceleration, Ball Control,
Marking etc. just assigns a number between 1 and 100 for that particular player depending on the player’s strength in that area.But,
descriptions will be needed for the player positions as it will not be familiar for the people who don’t follow soccer. So, following 
are the descriptions of the player positions that I have used in the dataset.

Abbreviation	Meaning
CB	Center Back
LB	Left Back
RB	Right Back
CM	Center Midfield
LM	Left Midfield
RM	Right Midfield
ST	Striker
LW	Left Wing
RW	Right Wing





```python
interesting_columns = [
    'Name', 
    'Age', 
    'Photo', 
    'Nationality', 
    'Overall', 
    'Potential', 
    'Club', 
    'Value', 
    'Wage', 
    'Preferred Positions'
]
dataset = pd.DataFrame(dataset, columns=interesting_columns)
```

Data preprocessing

Numeric columns of Value and Wage

Right away we can see that values in columns: 'Value' and 'Wage' aren't 
numeric but objects. Firstly we need to preprocess the data to make it usable for us. 
We will use short supporting function to convert values in those two columns into numbers. 
We will end up with two new columns 'ValueNum' and 'WageNum' that will contain numeric values.





```python
# Supporting function for converting string values into numbers
def str2number(amount):
    if amount[-1] == 'M':
        return float(amount[1:-1])*1000000
    elif amount[-1] == 'K':
        return float(amount[1:-1])*1000
    else:
        return float(amount[1:])
    

#dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
dataset['ValueNum'] = dataset['Value'].apply(lambda x: str2number(x))
dataset['WageNum'] = dataset['Wage'].apply(lambda x: str2number(x))
```


```python
#We will also add two additional columns: 'ValueCategory' and 'WageCategory'.
#We will use those collumns to divide players into ten classes basing on their value and incom.

max_value = float(dataset['ValueNum'].max() + 1)
max_wage = float(dataset['WageNum'].max() + 1)

# Supporting function for creating category columns 'ValueCategory' and 'WageCategory'
def mappingAmount(x, max_amount):
    for i in range(0, 10):
        if x >= max_amount/10*i and x < max_amount/10*(i+1):
            return i
        
dataset['ValueCategory'] = dataset['ValueNum'].apply(lambda x: mappingAmount(x, max_value))
dataset['WageCategory'] = dataset['WageNum'].apply(lambda x: mappingAmount(x, max_wage))
```


```python
#Potential points
dataset['PotentialPoints'] = dataset['Potential'] - dataset['Overall']
# Preferred position
dataset['Position'] = dataset['Preferred Positions'].str.split().str[0]
dataset['PositionNum'] = dataset['Preferred Positions'].apply(lambda x: len(x.split()))
## List of countries for each continent
continents = {
    'Africa' : ['Algeria','Angola','Benin','Botswana','Burkina','Burundi','Cameroon','Cape Verde','Central African Republic','Chad','Comoros','Congo','DR Congo','Djibouti','Egypt','Equatorial Guinea','Eritrea','Ethiopia','Gabon','Gambia','Ghana','Guinea','Guinea Bissau','Ivory Coast','Kenya','Lesotho','Liberia','Libya','Madagascar','Malawi','Mali','Mauritania','Mauritius','Morocco','Mozambique','Namibia','Niger','Nigeria','Rwanda','Sao Tome and Principe','Senegal','Seychelles','Sierra Leone','Somalia','South Africa','South Sudan','Sudan','Swaziland','Tanzania','Togo','Tunisia','Uganda','Zambia','Zimbabwe','Burkina Faso'],
    'Antarctica' : ['Fiji','Kiribati','Marshall Islands','Micronesia','Nauru','New Zealand','Palau','Papua New Guinea','Samoa','Solomon Islands','Tonga','Tuvalu','Vanuatu'],
    'Asia' : ['Afghanistan','Bahrain','Bangladesh','Bhutan','Brunei','Burma (Myanmar)','Cambodia','China','China PR','East Timor','India','Indonesia','Iran','Iraq','Israel','Japan','Jordan','Kazakhstan','North Korea','South Korea','Korea Republic','Korea DPR','Kuwait','Kyrgyzstan','Laos','Lebanon','Malaysia','Maldives','Mongolia','Nepal','Oman','Pakistan','Palestine','Philippines','Qatar','Russian Federation','Saudi Arabia','Singapore','Sri Lanka','Syria','Tajikistan','Thailand','Turkey','Turkmenistan','United Arab Emirates','Uzbekistan','Vietnam','Yemen','Russia'],
    'Australia Oceania' : ['Australia','New Caledonia'],
    'Europe' : ['Albania','Andorra','Armenia','Austria','Azerbaijan','Belarus','Belgium','Bosnia Herzegovina','Bulgaria','Croatia','Cyprus','Czech Republic','Denmark','Estonia','Finland','France','FYR Macedonia','Georgia','Germany','Greece','Hungary','Iceland','Ireland','Italy','Kosovo','Latvia','Liechtenstein','Lithuania','Luxembourg','Macedonia','Malta','Moldova','Monaco','Montenegro','Netherlands','Northern Ireland','Norway','Poland','Portugal','Romania','San Marino','Scotland','Serbia','Slovakia','Slovenia','Spain','Sweden','Switzerland','Ukraine','England','Vatican City','Republic of Ireland','Wales'],
    'North America' : ['Antigua and Barbuda','Bahamas','Barbados','Belize','Canada','Costa Rica','Cuba','Dominica','Dominican Republic','El Salvador','Grenada','Guatemala','Haiti','Honduras','Jamaica','Mexico','Nicaragua','Panama','Saint Kitts and Nevis','Saint Lucia','Saint Vincent and the Grenadines','Trinidad and Tobago','United States'],
    'South America' : ['Argentina','Bolivia','Brazil','Chile','Colombia','Curacao','Ecuador','Guyana','Paraguay','Peru','Suriname','Trinidad & Tobago','Uruguay','Venezuela']
}

# Function matching continent to countries
def find_continent(x, continents_list):
    # Iteration over 
    for key in continents_list:
        if x in continents_list[key]:
            return key
    return np.NaN

dataset['Continent'] = dataset['Nationality'].apply(lambda x: find_continent(x, continents))
```


```python
from IPython.core.display import display, HTML, Javascript
from string import Template
import json
import IPython.display
```


```python
top_1000 = dataset.sort_values("Overall", ascending=False).reset_index().head(1000)[["Name", "Nationality", "Continent", "Overall", "Club"]]
```


```python
Africa = top_1000[top_1000["Continent"]=='Africa']
Antarctica = top_1000[top_1000["Continent"]=='Antarctica']
Asia = top_1000[top_1000["Continent"]=='Asia']
Australia_Oceania =  top_1000[top_1000["Continent"]=='Australia_Oceania']
Europe = top_1000[top_1000["Continent"]=='Europe']
North_america = top_1000[top_1000["Continent"]=='North_america']
South_america = top_1000[top_1000["Continent"]=='South_america']

data = {}
data["name"] = "DISTRIBUTION OF TOP 1000 PLAERS DUE TO NATIONALITY"
data["children"] = []
# Split dataset into Continents:~
for continent in top_1000['Continent'].unique():
    
    continent_set = top_1000[top_1000["Continent"]==continent]
    continent_dict = {}
    continent_dict["name"] = continent
    continent_dict["children"] = []
    
    for country in continent_set['Nationality'].unique():
        
        countries_set = continent_set[continent_set['Nationality']==country][['Name', 'Overall']]
        
        country_dict = {}
        country_dict["name"] = country
        country_dict["children"] = []
        
        for player in countries_set.values:
            
            player_dict = {}
            player_dict['name'] = player[0]
            player_dict['size'] = player[1]
            country_dict["children"].append(player_dict)
            
        continent_dict['children'].append(country_dict)
        
    data["children"].append(continent_dict)

```


```python
North_america_dict = {}
North_america_dict['name'] = 'North_america'
North_america_dict['children'] = []
for country in North_america['Nationality'].unique():
    list_of_countries = North_america[North_america['Nationality']==country][['Name', 'Overall']].rename(columns={'Name': 'name', 'Overall': 'size'})
    tmp_dict = {}
    tmp_dict["name"] = country
    tmp_dict["children"] = []
    for row in list_of_countries.values:
        player_tmp = {}
        player_tmp['name'] = row[0]
        player_tmp['size'] = row[1]
        tmp_dict["children"].append(player_tmp)
    North_america_dict['children'].append(tmp_dict)
```


```python
html_string = """
<!DOCTYPE html>
<meta charset="utf-8">
<style>

.node {
  cursor: pointer;
}

.node:hover {
  stroke: #000;
  stroke-width: 1.5px;
}

.node--leaf {
  fill: white;
}

.label {
  font: 11px "Helvetica Neue", Helvetica, Arial, sans-serif;
  text-anchor: middle;
  text-shadow: 0 1px 0 #fff, 1px 0 0 #fff, -1px 0 0 #fff, 0 -1px 0 #fff;
}

.label,
.node--root,
.node--leaf {
  pointer-events: none;
}

</style>
<svg width="800" height="1000"></svg>
"""
```


```python
js_string="""
 require.config({
    paths: {
        d3: "https://d3js.org/d3.v4.min"
     }
 });

  require(["d3"], function(d3) {

   console.log(d3);

var svg = d3.select("svg"),
    margin = 20,
    diameter = +svg.attr("width"),
    g = svg.append("g").attr("transform", "translate(" + diameter / 2 + "," + diameter / 2 + ")");

var color = d3.scaleSequential(d3.interpolatePlasma)
    .domain([-4, 4]);

var pack = d3.pack()
    .size([diameter - margin, diameter - margin])
    .padding(2);

d3.json("output.json", function(error, root) {
  if (error) throw error;

  root = d3.hierarchy(root)
      .sum(function(d) { return d.size; })
      .sort(function(a, b) { return b.value - a.value; });

  var focus = root,
      nodes = pack(root).descendants(),
      view;

  var circle = g.selectAll("circle")
    .data(nodes)
    .enter().append("circle")
    .attr("class", function(d) { return d.parent ? d.children ? "node" : "node node--leaf" : "node node--root"; })
      .style("fill", function(d) { return d.children ? color(d.depth) : null; })
      .on("click", function(d) { if (focus !== d) zoom(d), d3.event.stopPropagation(); });

  var text = g.selectAll("text")
    .data(nodes)
    .enter().append("text")
      .attr("class", "label")
      .style("fill-opacity", function(d) { return d.parent === root ? 1 : 0; })
      .style("display", function(d) { return d.parent === root ? "inline" : "none"; })
      .text(function(d) { return d.data.name; });

  var node = g.selectAll("circle,text");

  svg
      .style("background", color(-1))
      .on("click", function() { zoom(root); });

  zoomTo([root.x, root.y, root.r * 2 + margin]);

  function zoom(d) {
    var focus0 = focus; focus = d;

    var transition = d3.transition()
        .duration(d3.event.altKey ? 7500 : 750)
        .tween("zoom", function(d) {
          var i = d3.interpolateZoom(view, [focus.x, focus.y, focus.r * 2 + margin]);
          return function(t) { zoomTo(i(t)); };
        });

    transition.selectAll("text")
      .filter(function(d) { return d.parent === focus || this.style.display === "inline"; })
        .style("fill-opacity", function(d) { return d.parent === focus ? 1 : 0; })
        .on("start", function(d) { if (d.parent === focus) this.style.display = "inline"; })
        .on("end", function(d) { if (d.parent !== focus) this.style.display = "none"; });
  }
  function zoomTo(v) {
    var k = diameter / v[2]; view = v;
    node.attr("transform", function(d) { return "translate(" + (d.x - v[0]) * k + "," + (d.y - v[1]) * k + ")"; });
    circle.attr("r", function(d) { return d.r * k; });
  }
});
  });
 """
```


```python
with open('output.json', 'w') as outfile:  
    json.dump(data, outfile)
```


```python
from IPython.display import display, HTML

display(HTML(data="""
<style>
    div#notebook-container    { height: 95%; }
    div#menubar-container     { height: 65%; }
    div#maintoolbar-container { height: 99%; }
</style>
"""))
```



<style>
    div#notebook-container    { height: 95%; }
    div#menubar-container     { height: 65%; }
    div#maintoolbar-container { height: 99%; }
</style>




```python
h = display(HTML(html_string))
j = IPython.display.Javascript(js_string)
IPython.display.display_javascript(j)

```



<!DOCTYPE html>
<meta charset="utf-8">
<style>

.node {
  cursor: pointer;
}

.node:hover {
  stroke: #000;
  stroke-width: 1.5px;
}

.node--leaf {
  fill: white;
}

.label {
  font: 11px "Helvetica Neue", Helvetica, Arial, sans-serif;
  text-anchor: middle;
  text-shadow: 0 1px 0 #fff, 1px 0 0 #fff, -1px 0 0 #fff, 0 -1px 0 #fff;
}

.label,
.node--root,
.node--leaf {
  pointer-events: none;
}

</style>
<svg width="800" height="1000"></svg>





Geographic distribution of players

Circle-packing is the arrangement of circles inside some demarcation so that none of the circles overlap. Circle packing also displays hierarchy where you can get smaller clusters of circles packed within a bigger circle which itself is arranged next to or within other circles. The D3.js plot will be interactive and dynamic, where one is able to invoke zoomable animations at different regions and clusters with the click of a mouse button. The D3.js plot will be interactive and dynamic, where one is able to invoke zoomable animations at different regions and clusters with the click of a mouse button. Each of the player’s nationality was mapped to its respective continent. There were 162 distinct Nationality values in the dataset and these countries were mapped to 6 continents: Asia, Europe, Africa, North America, South America and Australia/Oceania. In the plot, the 6 continents will be the parent class (outer circles). We can dive deeper within this class to find the countries (sub-class / sub-circles) and within each country, we will find the players (inner circles). The size of the player circle is determined by the Overall variable. A continent dictionary was created with the names of the continents as the keys and the list of countries as the values for each key. A function was defined to assign the continent for each country. The top 2000 players were chosen based on the overall value. Groupings of the players were hence identified using the Nationality and Continent. This grouping will be displayed with the circle graph plot and is fed into the json file. The data to be displayed is stored in the json file.

The FacetGrid
The core seaborn utility for faceting is the FacetGrid. A FacetGrid is an object which stores some information on how you want to break up your data visualization.

For example, suppose that we're interested in (as in the previous notebook) comparing strikers and goalkeepers in some way. To do this, we can create a FacetGrid with our data, telling it that we want to break the Position variable down by col (column).

Since we're zeroing in on just two positions in particular, this results in a pair of grids ready for us to "do" something with


```python
df = dataset

g = sns.FacetGrid(df, col="Position", col_wrap=6)
g.map(sns.kdeplot, "Overall")

```

    /Users/ashinbasheer/anaconda3/lib/python3.7/site-packages/scipy/stats/stats.py:1713: FutureWarning:
    
    Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
    





    <seaborn.axisgrid.FacetGrid at 0x1a1b2a16d8>




![png](output_24_2.png)



```python
df = dataset[dataset['Position'].isin(['ST', 'GK'])]
df = df[df['Club'].isin(['Real Madrid CF', 'FC Barcelona', 'Atlético Madrid'])]

g = sns.FacetGrid(df, row="Position", col="Club")
g.map(sns.violinplot, "Overall", order=[])

# Obs: 
# I didn't understand the "order" parameter here, but, a warning is displayed without it
# UserWarning: Using the violinplot function without specifying `order` is likely to produce an incorrect plot.
```




    <seaborn.axisgrid.FacetGrid at 0x1a1c0480f0>




![png](output_25_1.png)



```python
from plotly.offline import init_notebook_mode, iplot
#plotly
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected =True)
import plotly.graph_objs as go
from plotly import tools
import matplotlib.pyplot as plt     #matplotlib

```


        <script type="text/javascript">
        window.PlotlyConfig = {MathJaxConfig: 'local'};
        if (window.MathJax) {MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}
        if (typeof require !== 'undefined') {
        require.undef("plotly");
        requirejs.config({
            paths: {
                'plotly': ['https://cdn.plot.ly/plotly-latest.min']
            }
        });
        require(['plotly'], function(Plotly) {
            window._Plotly = Plotly;
        });
        }
        </script>
        


Club Rates of the Top 100 Players with Pie Plot


```python
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
```

# Club Rates of Top 100 Players


```python
#Prepare Data Frame
d_frame = dataset.iloc[:100,:]
donut= d_frame.Club.value_counts()
labels = d_frame.Club.value_counts().index

#Creat figure
fig = {
    "data":
    [
        {
            "values": donut,
            "labels": labels,
            "domain": {"x": [0, 1]},
            "name": "Clubs Rate",
            "hoverinfo": "label+percent+name",
            "hole": .4,
            "type": "pie"
        }, 
    ],
    "layout":
    {
        "title":"Club rates of the top 100 players",
        "annotations":
        [
            { 
                "font":{"size":20},
                "showarrow":False,
                "text": "",
                "x": 0,
                "y": 1
            },
        ]
    }
}
iplot(fig)
```


<div>
        
        
            <div id="8124a2a2-d160-4b7b-9f6a-d60f9bd37c46" class="plotly-graph-div" style="height:525px; width:100%;"></div>
            <script type="text/javascript">
                require(["plotly"], function(Plotly) {
                    window.PLOTLYENV=window.PLOTLYENV || {};
                    window.PLOTLYENV.BASE_URL='https://plot.ly';
                    
                if (document.getElementById("8124a2a2-d160-4b7b-9f6a-d60f9bd37c46")) {
                    Plotly.newPlot(
                        '8124a2a2-d160-4b7b-9f6a-d60f9bd37c46',
                        [{"domain": {"x": [0, 1]}, "hole": 0.4, "hoverinfo": "label+percent+name", "labels": ["Real Madrid CF", "FC Bayern Munich", "Juventus", "FC Barcelona", "Chelsea", "Manchester United", "Paris Saint-Germain", "Tottenham Hotspur", "Manchester City", "Atl\u00e9tico Madrid", "Arsenal", "Napoli", "Borussia Dortmund", "Liverpool", "Inter", "AS Monaco", "Bayer 04 Leverkusen", "Athletic Club de Bilbao", "Roma", "AS Saint-\u00c9tienne", "Be\u015fikta\u015f JK", "Milan"], "name": "Clubs Rate", "type": "pie", "uid": "8b943215-e1f1-419f-9b6f-a86db88c2ffe", "values": [13, 12, 9, 8, 7, 6, 6, 6, 6, 5, 4, 3, 3, 2, 2, 2, 1, 1, 1, 1, 1, 1]}],
                        {"annotations": [{"font": {"size": 20}, "showarrow": false, "text": "", "x": 0, "y": 1}], "title": {"text": "Club rates of the top 100 players"}},
                        {"showLink": false, "linkText": "Export to plot.ly", "plotlyServerURL": "https://plot.ly", "responsive": true}
                    ).then(function(){
                            
var gd = document.getElementById('8124a2a2-d160-4b7b-9f6a-d60f9bd37c46');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };
                });
            </script>
        </div>


 Top 10 Players 'Ball Control', 'Dribbling' and 'Shot Power' Features Comparison with Scatter 3D Plot


```python
#Now let's see the top 10 countries with the have more players
dataset.groupby("Nationality").Name.count().sort_values(ascending=False).head(10)
```




    Nationality
    England        1630
    Germany        1140
    Spain          1019
    France          978
    Argentina       965
    Brazil          812
    Italy           799
    Colombia        592
    Japan           469
    Netherlands     429
    Name: Name, dtype: int64



# Map Distribution of the Players and How many players from the same Country?


```python
#How many players are from the same country?
df = dataset['Nationality'].value_counts()

iplot([
    go.Choropleth(
    locationmode='country names',
    locations=df.index.values,
    text= df.index,
    z=df.values,
    colorscale= 'Jet'
    )
])
```


<div>
        
        
            <div id="72a30eda-3fd5-4594-9082-c6ddbb663281" class="plotly-graph-div" style="height:525px; width:100%;"></div>
            <script type="text/javascript">
                require(["plotly"], function(Plotly) {
                    window.PLOTLYENV=window.PLOTLYENV || {};
                    window.PLOTLYENV.BASE_URL='https://plot.ly';
                    
                if (document.getElementById("72a30eda-3fd5-4594-9082-c6ddbb663281")) {
                    Plotly.newPlot(
                        '72a30eda-3fd5-4594-9082-c6ddbb663281',
                        [{"colorscale": "Jet", "locationmode": "country names", "locations": ["England", "Germany", "Spain", "France", "Argentina", "Brazil", "Italy", "Colombia", "Japan", "Netherlands", "Republic of Ireland", "United States", "Chile", "Sweden", "Portugal", "Mexico", "Denmark", "Poland", "Norway", "Korea Republic", "Saudi Arabia", "Russia", "Scotland", "Turkey", "Belgium", "Austria", "Switzerland", "Australia", "Uruguay", "Serbia", "Senegal", "Nigeria", "Wales", "Ghana", "Croatia", "Ivory Coast", "Greece", "Cameroon", "Northern Ireland", "Morocco", "South Africa", "Czech Republic", "Paraguay", "Slovenia", "Slovakia", "Finland", "Romania", "DR Congo", "Algeria", "Bosnia Herzegovina", "Canada", "Iceland", "Venezuela", "Ukraine", "Mali", "New Zealand", "Jamaica", "Hungary", "Ecuador", "Albania", "Bulgaria", "Egypt", "Kosovo", "Georgia", "Tunisia", "China PR", "India", "Bolivia", "Peru", "Costa Rica", "Guinea", "Montenegro", "Cape Verde", "Congo", "FYR Macedonia", "Iran", "Guinea Bissau", "Panama", "Angola", "Gambia", "Armenia", "Haiti", "Burkina Faso", "Benin", "Honduras", "Lithuania", "Israel", "Gabon", "Cyprus", "Zimbabwe", "Curacao", "Comoros", "Belarus", "Sierra Leone", "Trinidad & Tobago", "Iraq", "Estonia", "Togo", "Luxembourg", "Latvia", "Zambia", "Kenya", "Azerbaijan", "Faroe Islands", "Korea DPR", "Moldova", "Qatar", "Syria", "Equatorial Guinea", "Uzbekistan", "Antigua & Barbuda", "Guyana", "Lebanon", "Central African Rep.", "Madagascar", "Palestine", "Montserrat", "Bermuda", "Uganda", "Philippines", "Malta", "Kazakhstan", "Niger", "Afghanistan", "Mozambique", "Dominican Republic", "Suriname", "St Kitts Nevis", "Liechtenstein", "Tanzania", "Gibraltar", "Sudan", "Namibia", "Cuba", "Mauritania", "Kuwait", "Liberia", "Chad", "El Salvador", "Thailand", "Libya", "New Caledonia", "Puerto Rico", "Ethiopia", "Mauritius", "Swaziland", "Guam", "Vietnam", "Kyrgyzstan", "Oman", "Sri Lanka", "Fiji", "Somalia", "Barbados", "Burundi", "Eritrea", "S\u00e3o Tom\u00e9 & Pr\u00edncipe", "San Marino", "St Lucia", "Turkmenistan", "Hong Kong", "Brunei Darussalam", "Belize", "Guatemala", "Grenada"], "text": ["England", "Germany", "Spain", "France", "Argentina", "Brazil", "Italy", "Colombia", "Japan", "Netherlands", "Republic of Ireland", "United States", "Chile", "Sweden", "Portugal", "Mexico", "Denmark", "Poland", "Norway", "Korea Republic", "Saudi Arabia", "Russia", "Scotland", "Turkey", "Belgium", "Austria", "Switzerland", "Australia", "Uruguay", "Serbia", "Senegal", "Nigeria", "Wales", "Ghana", "Croatia", "Ivory Coast", "Greece", "Cameroon", "Northern Ireland", "Morocco", "South Africa", "Czech Republic", "Paraguay", "Slovenia", "Slovakia", "Finland", "Romania", "DR Congo", "Algeria", "Bosnia Herzegovina", "Canada", "Iceland", "Venezuela", "Ukraine", "Mali", "New Zealand", "Jamaica", "Hungary", "Ecuador", "Albania", "Bulgaria", "Egypt", "Kosovo", "Georgia", "Tunisia", "China PR", "India", "Bolivia", "Peru", "Costa Rica", "Guinea", "Montenegro", "Cape Verde", "Congo", "FYR Macedonia", "Iran", "Guinea Bissau", "Panama", "Angola", "Gambia", "Armenia", "Haiti", "Burkina Faso", "Benin", "Honduras", "Lithuania", "Israel", "Gabon", "Cyprus", "Zimbabwe", "Curacao", "Comoros", "Belarus", "Sierra Leone", "Trinidad & Tobago", "Iraq", "Estonia", "Togo", "Luxembourg", "Latvia", "Zambia", "Kenya", "Azerbaijan", "Faroe Islands", "Korea DPR", "Moldova", "Qatar", "Syria", "Equatorial Guinea", "Uzbekistan", "Antigua & Barbuda", "Guyana", "Lebanon", "Central African Rep.", "Madagascar", "Palestine", "Montserrat", "Bermuda", "Uganda", "Philippines", "Malta", "Kazakhstan", "Niger", "Afghanistan", "Mozambique", "Dominican Republic", "Suriname", "St Kitts Nevis", "Liechtenstein", "Tanzania", "Gibraltar", "Sudan", "Namibia", "Cuba", "Mauritania", "Kuwait", "Liberia", "Chad", "El Salvador", "Thailand", "Libya", "New Caledonia", "Puerto Rico", "Ethiopia", "Mauritius", "Swaziland", "Guam", "Vietnam", "Kyrgyzstan", "Oman", "Sri Lanka", "Fiji", "Somalia", "Barbados", "Burundi", "Eritrea", "S\u00e3o Tom\u00e9 & Pr\u00edncipe", "San Marino", "St Lucia", "Turkmenistan", "Hong Kong", "Brunei Darussalam", "Belize", "Guatemala", "Grenada"], "type": "choropleth", "uid": "e3ad8415-ec2b-4ab5-8df7-438ffbf671c9", "z": [1630, 1140, 1019, 978, 965, 812, 799, 592, 469, 429, 417, 381, 375, 368, 367, 360, 346, 337, 333, 330, 329, 306, 300, 291, 272, 266, 233, 227, 153, 133, 129, 126, 123, 117, 109, 101, 98, 86, 86, 78, 77, 73, 69, 65, 64, 62, 59, 58, 57, 55, 55, 53, 51, 49, 46, 38, 37, 37, 37, 35, 33, 32, 32, 31, 31, 30, 30, 30, 30, 29, 25, 25, 22, 22, 17, 17, 16, 16, 15, 15, 14, 14, 14, 14, 13, 12, 12, 12, 11, 11, 11, 9, 9, 9, 9, 9, 8, 8, 8, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}],
                        {},
                        {"showLink": false, "linkText": "Export to plot.ly", "plotlyServerURL": "https://plot.ly", "responsive": true}
                    ).then(function(){
                            
var gd = document.getElementById('72a30eda-3fd5-4594-9082-c6ddbb663281');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };
                });
            </script>
        </div>


Nationality Rates of the Top 100 Players with Pie Plot

# Donut Chart for Nationality rate of Top 100 players accoding to thir Club rate


```python
#Prepare Data Frame
d_frame = dataset.iloc[:100,:]
donut= d_frame.Nationality.value_counts()
labels = d_frame.Nationality.value_counts().index #Country names of the top 100 players

#Creat Figure
fig = {
    "data":
    [
        {
            "values": donut,
            "labels": labels,
            "domain": {"x": [0, 1]},
            "name": "Clubs Rate",
            "hoverinfo": "label+percent+name",
            "hole": .4,
            "type": "pie"
        }, 
    ],
    "layout":
    {
        "title":"Nationality rates of the top 100 players",
        "annotations":
        [
            { 
                "font":{"size":20},
                "showarrow":False,
                "text": "",
                "x": 0,
                "y": 1
            },
        ]
    }
}
iplot(fig)

```


<div>
        
        
            <div id="a64870af-84d4-43c6-9047-671b0df58af4" class="plotly-graph-div" style="height:525px; width:100%;"></div>
            <script type="text/javascript">
                require(["plotly"], function(Plotly) {
                    window.PLOTLYENV=window.PLOTLYENV || {};
                    window.PLOTLYENV.BASE_URL='https://plot.ly';
                    
                if (document.getElementById("a64870af-84d4-43c6-9047-671b0df58af4")) {
                    Plotly.newPlot(
                        'a64870af-84d4-43c6-9047-671b0df58af4',
                        [{"domain": {"x": [0, 1]}, "hole": 0.4, "hoverinfo": "label+percent+name", "labels": ["Spain", "Germany", "France", "Belgium", "Brazil", "Italy", "Argentina", "Croatia", "Portugal", "Uruguay", "Slovenia", "Poland", "Chile", "England", "Ivory Coast", "Gabon", "Colombia", "Czech Republic", "Wales", "Bosnia Herzegovina", "Costa Rica", "Denmark", "Slovakia", "Austria", "Netherlands", "Armenia", "Greece", "Sweden", "Senegal"], "name": "Clubs Rate", "type": "pie", "uid": "6dcc3266-d202-4fe5-88e1-11a96c1e15e6", "values": [15, 11, 11, 10, 9, 7, 5, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}],
                        {"annotations": [{"font": {"size": 20}, "showarrow": false, "text": "", "x": 0, "y": 1}], "title": {"text": "Nationality rates of the top 100 players"}},
                        {"showLink": false, "linkText": "Export to plot.ly", "plotlyServerURL": "https://plot.ly", "responsive": true}
                    ).then(function(){
                            
var gd = document.getElementById('a64870af-84d4-43c6-9047-671b0df58af4');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };
                });
            </script>
        </div>



```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pylab import rcParams

from plotly.offline import iplot, init_notebook_mode
from geopy.geocoders import Nominatim
import missingno as msno
import plotly.plotly as py

%matplotlib inline
rcParams["figure.figsize"] = 12, 8
rcParams["font.size"] = 3
rcParams["xtick.labelsize"] = 12
rcParams["ytick.labelsize"] = 12

```

# Distribution of the Overall Variable 


```python
# Overall ratings
fig, axe = plt.subplots()
ax = sns.distplot(dataset["Overall"], bins=len(dataset["Overall"].value_counts().index))
ax.set_xticks(range(0, 100, 5))
ax.set_xlim(0,100)
ax.set_xlabel("Overall Rating",fontsize=15)
fig.tight_layout()
```


![png](output_40_0.png)



```python
import pandas as pd
from math import pi
import matplotlib.pyplot as plt
%matplotlib inline

#Create a data frame from Messi and Ronaldo's 6 Ultimate Team data points from FIFA 18
Messi = {'Pace':89,'Shooting':90,'Passing':86,'Dribbling':95,'Defending':26,'Physical':61}
Ronaldo = {'Pace':90,'Shooting':93,'Passing':82,'Dribbling':90,'Defending':33,'Physical':80}

data = pd.DataFrame([Messi,Ronaldo], index = ["Messi","Ronaldo"])
```


```python
data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Defending</th>
      <th>Dribbling</th>
      <th>Pace</th>
      <th>Passing</th>
      <th>Physical</th>
      <th>Shooting</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Messi</th>
      <td>26</td>
      <td>95</td>
      <td>89</td>
      <td>86</td>
      <td>61</td>
      <td>90</td>
    </tr>
    <tr>
      <th>Ronaldo</th>
      <td>33</td>
      <td>90</td>
      <td>90</td>
      <td>82</td>
      <td>80</td>
      <td>93</td>
    </tr>
  </tbody>
</table>
</div>




```python
Attributes =list(data)
AttNo = len(Attributes)
```


```python
values = data.iloc[1].tolist()
values += values [:1]
values
```




    [33, 90, 90, 82, 80, 93, 33]




```python
angles = [n / float(AttNo) * 2 * pi for n in range(AttNo)]
angles += angles [:1]
```

# Radar Chart for Ronaldo and their Attributes


```python
ax = plt.subplot(111, polar=True)

#Add the attribute labels to our axes
plt.xticks(angles[:-1],Attributes)

#Plot the line around the outside of the filled area, using the angles and values calculated before
ax.plot(angles,values)

#Fill in the area plotted in the last line
ax.fill(angles, values, 'teal', alpha=0.1)

#Give the plot a title and show it
ax.set_title("Ronaldo")
plt.show()
```


![png](output_47_0.png)


# Radar Chart For Comparison of Messi Vs. Ronaldo


```python
#Find the values and angles for Messi - from the table at the top of the page
values2 = data.iloc[0].tolist()
values2 += values2 [:1]

angles2 = [n / float(AttNo) * 2 * pi for n in range(AttNo)]
angles2 += angles2 [:1]


#Create the chart as before, but with both Ronaldo's and Messi's angles/values
ax = plt.subplot(111, polar=True)

plt.xticks(angles[:-1],Attributes)

ax.plot(angles,values)
ax.fill(angles, values, 'teal', alpha=0.1)

ax.plot(angles2,values2)
ax.fill(angles2, values2, 'red', alpha=0.1)

#Rather than use a title, individual text points are added
plt.figtext(0.2,0.9,"Messi",color="red")
plt.figtext(0.2,0.85,"v")
plt.figtext(0.2,0.8,"Ronaldo",color="teal")
plt.show()
```


![png](output_49_0.png)


Radar charts are an interesting way to display data and allow us to compare two observations quite nicely. In this, we have used them to compare fictional FIFA players, but analysts have used this format very innovatively to display actual performance data in an engaging format.


```python
from wordcloud import WordCloud
```

# Names of the Top 100 Players with 'Word Cloud


```python

#Prepare Data Frame
d_frame = dataset.Name[:50]

plt.subplots(figsize=(10,10))
wordcloud = WordCloud(
                   background_color='White',
                        width = 700,
                        height = 400
    ).generate(" ".join(d_frame))

plt.imshow(wordcloud)
plt.axis('off')
plt.savefig('graph.png')
```


![png](output_53_0.png)



```python
dataset=pd.read_csv("https://raw.githubusercontent.com/fifa-players/Fifa/master/CompleteDataset.csv")
fifa_data=dataset
fifa_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>Name</th>
      <th>Age</th>
      <th>Photo</th>
      <th>Nationality</th>
      <th>Flag</th>
      <th>Overall</th>
      <th>Potential</th>
      <th>Club</th>
      <th>Club Logo</th>
      <th>...</th>
      <th>RB</th>
      <th>RCB</th>
      <th>RCM</th>
      <th>RDM</th>
      <th>RF</th>
      <th>RM</th>
      <th>RS</th>
      <th>RW</th>
      <th>RWB</th>
      <th>ST</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Cristiano Ronaldo</td>
      <td>32</td>
      <td>https://cdn.sofifa.org/48/18/players/20801.png</td>
      <td>Portugal</td>
      <td>https://cdn.sofifa.org/flags/38.png</td>
      <td>94</td>
      <td>94</td>
      <td>Real Madrid CF</td>
      <td>https://cdn.sofifa.org/24/18/teams/243.png</td>
      <td>...</td>
      <td>61.0</td>
      <td>53.0</td>
      <td>82.0</td>
      <td>62.0</td>
      <td>91.0</td>
      <td>89.0</td>
      <td>92.0</td>
      <td>91.0</td>
      <td>66.0</td>
      <td>92.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>L. Messi</td>
      <td>30</td>
      <td>https://cdn.sofifa.org/48/18/players/158023.png</td>
      <td>Argentina</td>
      <td>https://cdn.sofifa.org/flags/52.png</td>
      <td>93</td>
      <td>93</td>
      <td>FC Barcelona</td>
      <td>https://cdn.sofifa.org/24/18/teams/241.png</td>
      <td>...</td>
      <td>57.0</td>
      <td>45.0</td>
      <td>84.0</td>
      <td>59.0</td>
      <td>92.0</td>
      <td>90.0</td>
      <td>88.0</td>
      <td>91.0</td>
      <td>62.0</td>
      <td>88.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>Neymar</td>
      <td>25</td>
      <td>https://cdn.sofifa.org/48/18/players/190871.png</td>
      <td>Brazil</td>
      <td>https://cdn.sofifa.org/flags/54.png</td>
      <td>92</td>
      <td>94</td>
      <td>Paris Saint-Germain</td>
      <td>https://cdn.sofifa.org/24/18/teams/73.png</td>
      <td>...</td>
      <td>59.0</td>
      <td>46.0</td>
      <td>79.0</td>
      <td>59.0</td>
      <td>88.0</td>
      <td>87.0</td>
      <td>84.0</td>
      <td>89.0</td>
      <td>64.0</td>
      <td>84.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>L. Suárez</td>
      <td>30</td>
      <td>https://cdn.sofifa.org/48/18/players/176580.png</td>
      <td>Uruguay</td>
      <td>https://cdn.sofifa.org/flags/60.png</td>
      <td>92</td>
      <td>92</td>
      <td>FC Barcelona</td>
      <td>https://cdn.sofifa.org/24/18/teams/241.png</td>
      <td>...</td>
      <td>64.0</td>
      <td>58.0</td>
      <td>80.0</td>
      <td>65.0</td>
      <td>88.0</td>
      <td>85.0</td>
      <td>88.0</td>
      <td>87.0</td>
      <td>68.0</td>
      <td>88.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>M. Neuer</td>
      <td>31</td>
      <td>https://cdn.sofifa.org/48/18/players/167495.png</td>
      <td>Germany</td>
      <td>https://cdn.sofifa.org/flags/21.png</td>
      <td>92</td>
      <td>92</td>
      <td>FC Bayern Munich</td>
      <td>https://cdn.sofifa.org/24/18/teams/21.png</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 75 columns</p>
</div>




```python
# In order to be able to start our ranking from 1, we have assigned the 'array' variable.
fifa_data['array'] = fifa_data['Unnamed: 0']+1
```


```python
# Let's change the names of some columns
fifa_data.rename(columns={'Ball control':'ball_control','Free kick accuracy':'free_kick',
                     'Shot power':'shot_power'}, inplace=True)
# Just take the columns we'll use
fifa_data= fifa_data[['array','Name','Age','Nationality','Overall','Potential','Club','Value',
                      'Wage','Special','Acceleration','ball_control','Dribbling','free_kick',
                      'Penalties','shot_power']]
```


```python
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#plotly
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected =True)
import plotly.graph_objs as go
from plotly import tools
from wordcloud import WordCloud     #word cloud library
import matplotlib.pyplot as plt     #matplotlib
```


        <script type="text/javascript">
        window.PlotlyConfig = {MathJaxConfig: 'local'};
        if (window.MathJax) {MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}
        if (typeof require !== 'undefined') {
        require.undef("plotly");
        requirejs.config({
            paths: {
                'plotly': ['https://cdn.plot.ly/plotly-latest.min']
            }
        });
        require(['plotly'], function(Plotly) {
            window._Plotly = Plotly;
        });
        }
        </script>
        


# Comparing the 'Overall' and 'Potential' of Players with Scatter Plot


```python
# Prepare Data Frame
d_frame = fifa_data.iloc[:100,:]

#Creating trace1
trace1 = go.Scatter(
    x= d_frame.array,
    y= d_frame.Overall,
    mode='lines',
    name='Overall',
    marker= dict(color='rgba(12, 255, 250,0.9)'),
    text= d_frame.Name
)
#Creating trace2
trace2 = go.Scatter(
    x= d_frame.array,
    y= d_frame.Potential,
    mode='lines+markers',
    name='Potential',
    
    text= d_frame.Name
)
data = [trace1,trace2]
layout= dict(title="Comparing the 'Overall' and 'Potential' of players",
            xaxis= dict(title='Player Rank', ticklen=5, zeroline=False)
            )
fig= dict(data=data, layout=layout)
iplot(fig)
```


<div>
        
        
            <div id="8ee6cae5-873c-420e-bce1-e6a14e98cf5a" class="plotly-graph-div" style="height:525px; width:100%;"></div>
            <script type="text/javascript">
                require(["plotly"], function(Plotly) {
                    window.PLOTLYENV=window.PLOTLYENV || {};
                    window.PLOTLYENV.BASE_URL='https://plot.ly';
                    
                if (document.getElementById("8ee6cae5-873c-420e-bce1-e6a14e98cf5a")) {
                    Plotly.newPlot(
                        '8ee6cae5-873c-420e-bce1-e6a14e98cf5a',
                        [{"marker": {"color": "rgba(12, 255, 250,0.9)"}, "mode": "lines", "name": "Overall", "text": ["Cristiano Ronaldo", "L. Messi", "Neymar", "L. Su\u00e1rez", "M. Neuer", "R. Lewandowski", "De Gea", "E. Hazard", "T. Kroos", "G. Higua\u00edn", "Sergio Ramos", "K. De Bruyne", "T. Courtois", "A. S\u00e1nchez", "L. Modri\u0107", "G. Bale", "S. Ag\u00fcero", "G. Chiellini", "G. Buffon", "P. Dybala", "J. Oblak", "A. Griezmann", "Thiago", "P. Aubameyang", "L. Bonucci", "J. Boateng", "D. God\u00edn", "M. Hummels", "M. \u00d6zil", "H. Lloris", "Thiago Silva", "Z. Ibrahimovi\u0107", "A. Robben", "N. Kant\u00e9", "M. Verratti", "P. Pogba", "C. Eriksen", "A. Vidal", "E. Cavani", "Marcelo", "M. Ham\u0161\u00edk", "I. Rakiti\u0107", "David Silva", "S. Handanovi\u010d", "Piqu\u00e9", "Iniesta", "H. Kane", "J. Rodr\u00edguez", "Isco", "D. Alaba", "R. Lukaku", "Alex Sandro", "T. M\u00fcller", "Sergio Busquets", "Coutinho", "M. Reus", "T. Alderweireld", "David Luiz", "Diego Costa", "R. Nainggolan", "Javi Mart\u00ednez", "D. Mertens", "Sokratis", "Miranda", "K. Benzema", "Cesc F\u00e0bregas", "F. Rib\u00e9ry", "Pepe", "P. \u010cech", "Y. Carrasco", "R. Varane", "Casemiro", "L. Insigne", "A. Lacazette", "K. Navas", "H. Mkhitaryan", "D. Suba\u0161i\u0107", "B. Leno", "M. ter Stegen", "K. Glik", "Jordi Alba", "I. G\u00fcndo\u011fan", "Azpilicueta", "A. Di Mar\u00eda", "M. Pjani\u0107", "C. Marchisio", "J. Vertonghen", "B. Matuidi", "S. Ruffier", "Filipe Lu\u00eds", "V. Kompany", "A. Barzagli", "E. Bailly", "Marco Asensio", "Bernardo Silva", "A. Laporte", "D. Alli", "S. Man\u00e9", "Carvajal", "J. Draxler"], "type": "scatter", "uid": "06ea671a-028c-4980-b628-cce44aa34934", "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100], "y": [94, 93, 92, 92, 92, 91, 90, 90, 90, 90, 90, 89, 89, 89, 89, 89, 89, 89, 89, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 87, 87, 87, 87, 87, 87, 87, 87, 87, 87, 87, 87, 87, 86, 86, 86, 86, 86, 86, 86, 86, 86, 86, 86, 86, 86, 86, 86, 86, 86, 86, 86, 86, 86, 86, 86, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 84, 84, 84, 84, 84, 84, 84, 84]}, {"mode": "lines+markers", "name": "Potential", "text": ["Cristiano Ronaldo", "L. Messi", "Neymar", "L. Su\u00e1rez", "M. Neuer", "R. Lewandowski", "De Gea", "E. Hazard", "T. Kroos", "G. Higua\u00edn", "Sergio Ramos", "K. De Bruyne", "T. Courtois", "A. S\u00e1nchez", "L. Modri\u0107", "G. Bale", "S. Ag\u00fcero", "G. Chiellini", "G. Buffon", "P. Dybala", "J. Oblak", "A. Griezmann", "Thiago", "P. Aubameyang", "L. Bonucci", "J. Boateng", "D. God\u00edn", "M. Hummels", "M. \u00d6zil", "H. Lloris", "Thiago Silva", "Z. Ibrahimovi\u0107", "A. Robben", "N. Kant\u00e9", "M. Verratti", "P. Pogba", "C. Eriksen", "A. Vidal", "E. Cavani", "Marcelo", "M. Ham\u0161\u00edk", "I. Rakiti\u0107", "David Silva", "S. Handanovi\u010d", "Piqu\u00e9", "Iniesta", "H. Kane", "J. Rodr\u00edguez", "Isco", "D. Alaba", "R. Lukaku", "Alex Sandro", "T. M\u00fcller", "Sergio Busquets", "Coutinho", "M. Reus", "T. Alderweireld", "David Luiz", "Diego Costa", "R. Nainggolan", "Javi Mart\u00ednez", "D. Mertens", "Sokratis", "Miranda", "K. Benzema", "Cesc F\u00e0bregas", "F. Rib\u00e9ry", "Pepe", "P. \u010cech", "Y. Carrasco", "R. Varane", "Casemiro", "L. Insigne", "A. Lacazette", "K. Navas", "H. Mkhitaryan", "D. Suba\u0161i\u0107", "B. Leno", "M. ter Stegen", "K. Glik", "Jordi Alba", "I. G\u00fcndo\u011fan", "Azpilicueta", "A. Di Mar\u00eda", "M. Pjani\u0107", "C. Marchisio", "J. Vertonghen", "B. Matuidi", "S. Ruffier", "Filipe Lu\u00eds", "V. Kompany", "A. Barzagli", "E. Bailly", "Marco Asensio", "Bernardo Silva", "A. Laporte", "D. Alli", "S. Man\u00e9", "Carvajal", "J. Draxler"], "type": "scatter", "uid": "6bf1d559-107e-4b67-95b2-0acd4d11ea2d", "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100], "y": [94, 93, 94, 92, 92, 91, 92, 91, 90, 90, 90, 92, 92, 89, 89, 89, 89, 89, 89, 93, 93, 91, 90, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 90, 91, 92, 91, 87, 87, 87, 87, 87, 87, 87, 87, 87, 90, 89, 90, 88, 90, 88, 86, 86, 89, 86, 87, 86, 86, 86, 86, 86, 86, 86, 86, 86, 86, 86, 86, 90, 92, 89, 86, 88, 85, 85, 85, 88, 89, 85, 85, 87, 87, 85, 86, 85, 85, 85, 85, 85, 85, 85, 89, 92, 91, 89, 90, 87, 87, 87]}],
                        {"title": {"text": "Comparing the 'Overall' and 'Potential' of players"}, "xaxis": {"ticklen": 5, "title": {"text": "Player Rank"}, "zeroline": false}},
                        {"showLink": false, "linkText": "Export to plot.ly", "plotlyServerURL": "https://plot.ly", "responsive": true}
                    ).then(function(){
                            
var gd = document.getElementById('8ee6cae5-873c-420e-bce1-e6a14e98cf5a');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };
                });
            </script>
        </div>


We are visualizing the trends of the Overall and potenial on the basis of the player's rank. So, the players are having high potenial as comapre to their overall value.

# Comparing 'Overall' and 'Potential' Values of Players between 90 and 100 with Bar Plot


```python
# Prepare Data Frame
d_frame = fifa_data.iloc[90:100,:]

#create trace1
trace1 = go.Bar(
    x= d_frame.array,
    y= d_frame.Overall,
    name= 'Overall',
    marker= dict(color= 'rgba(255,106,0,0.9)',
                line= dict(color= 'rgb(0,0,0)', width=1)),
    text= d_frame.Name
)
#Create trace2
trace2 = go.Bar(
    x= d_frame.array,
    y= d_frame.Potential,
    name= 'Potential',
    marker= dict(color= 'rgba(148, 255, 130,0.9)',
                line= dict(color='rgb(0,0,0)', width=1)),
)
data= [trace1, trace2]
layout= go.Layout(barmode= "group")
fig= go.Figure(data=data, layout=layout)
iplot(fig)
```


<div>
        
        
            <div id="0f5e2a4c-ef16-42ff-bbd6-257ea29ae9db" class="plotly-graph-div" style="height:525px; width:100%;"></div>
            <script type="text/javascript">
                require(["plotly"], function(Plotly) {
                    window.PLOTLYENV=window.PLOTLYENV || {};
                    window.PLOTLYENV.BASE_URL='https://plot.ly';
                    
                if (document.getElementById("0f5e2a4c-ef16-42ff-bbd6-257ea29ae9db")) {
                    Plotly.newPlot(
                        '0f5e2a4c-ef16-42ff-bbd6-257ea29ae9db',
                        [{"marker": {"color": "rgba(255,106,0,0.9)", "line": {"color": "rgb(0,0,0)", "width": 1}}, "name": "Overall", "text": ["V. Kompany", "A. Barzagli", "E. Bailly", "Marco Asensio", "Bernardo Silva", "A. Laporte", "D. Alli", "S. Man\u00e9", "Carvajal", "J. Draxler"], "type": "bar", "uid": "cf2d5e8e-8a3e-417d-8ae2-da4699f0f5cb", "x": [91, 92, 93, 94, 95, 96, 97, 98, 99, 100], "y": [85, 85, 84, 84, 84, 84, 84, 84, 84, 84]}, {"marker": {"color": "rgba(148, 255, 130,0.9)", "line": {"color": "rgb(0,0,0)", "width": 1}}, "name": "Potential", "type": "bar", "uid": "cbd16f8c-8351-49a3-8c1e-c81d51e9d4b5", "x": [91, 92, 93, 94, 95, 96, 97, 98, 99, 100], "y": [85, 85, 89, 92, 91, 89, 90, 87, 87, 87]}],
                        {"barmode": "group"},
                        {"showLink": false, "linkText": "Export to plot.ly", "plotlyServerURL": "https://plot.ly", "responsive": true}
                    ).then(function(){
                            
var gd = document.getElementById('0f5e2a4c-ef16-42ff-bbd6-257ea29ae9db');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };
                });
            </script>
        </div>


From above barplot, we are trying to do some analysis on the players from 90 to 100 in respect of overall and their potential values.Player no.91 and 92 have equal overal and potential but from player 93 to 100 everbody have the higher potenial value as compare to the overall value.

# Top 5 Players Free kick and Penalty Strokes Comparison with Bar Plot


```python
#Prepare Data Frame
d_frame = fifa_data.iloc[:5,:]

#Create trace1
trace1= {
    'x': d_frame.array,
    'y': d_frame.free_kick,
    'name': 'Free Kick',
    'type': 'bar',
    'text': d_frame.Name
};
#Create trace2
trace2= {
    'x': d_frame.array,
    'y': d_frame.Penalties,
    'name': 'Penalty',
    'type': 'bar',
    'text': d_frame.Name,
    'marker': dict(color= 'rgba(148, 255, 130,0.9)'),
};
data= [trace1, trace2]
layout= {
    'xaxis': {'title':'First 5 Player'},
    'barmode': 'relative',
    'title': 'Top 5 Players free kick and penalty strokes comparison'
};
fig = go.Figure(data=data, layout=layout)
iplot(fig)
```


<div>
        
        
            <div id="a9c5130c-f790-4bd5-84e5-33e4b23852fd" class="plotly-graph-div" style="height:525px; width:100%;"></div>
            <script type="text/javascript">
                require(["plotly"], function(Plotly) {
                    window.PLOTLYENV=window.PLOTLYENV || {};
                    window.PLOTLYENV.BASE_URL='https://plot.ly';
                    
                if (document.getElementById("a9c5130c-f790-4bd5-84e5-33e4b23852fd")) {
                    Plotly.newPlot(
                        'a9c5130c-f790-4bd5-84e5-33e4b23852fd',
                        [{"name": "Free Kick", "text": ["Cristiano Ronaldo", "L. Messi", "Neymar", "L. Su\u00e1rez", "M. Neuer"], "type": "bar", "uid": "f6d17461-cbcb-451c-9c20-94f755ab5170", "x": [1, 2, 3, 4, 5], "y": ["76", "90", "84", "84", "11"]}, {"marker": {"color": "rgba(148, 255, 130,0.9)"}, "name": "Penalty", "text": ["Cristiano Ronaldo", "L. Messi", "Neymar", "L. Su\u00e1rez", "M. Neuer"], "type": "bar", "uid": "572bc06c-c2b8-4baf-9da1-747dae3b4e43", "x": [1, 2, 3, 4, 5], "y": ["85", "74", "81", "85", "47"]}],
                        {"barmode": "relative", "title": {"text": "Top 5 Players free kick and penalty strokes comparison"}, "xaxis": {"title": {"text": "First 5 Player"}}},
                        {"showLink": false, "linkText": "Export to plot.ly", "plotlyServerURL": "https://plot.ly", "responsive": true}
                    ).then(function(){
                            
var gd = document.getElementById('a9c5130c-f790-4bd5-84e5-33e4b23852fd');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };
                });
            </script>
        </div>


From above barplot, Neymar and Suarez are aproximately same free kick and penalty but messi has high free kick but less penalty.Ronaldo has 85 penalty and 75 free kick.


```python
#from simpleplotly import FigureBuilder, FigureHolder
```

# Top 10 Players 'Ball Control', 'Dribbling' and 'Shot Power' Features Comparison with Scatter 3D Plot


```python
#Prepare Data Frame
d_frame = fifa_data.iloc[:10,:]

#Create trace
trace = go.Scatter3d(
    x=d_frame.ball_control,
    y=d_frame.Dribbling,
    z=d_frame.shot_power,
    text= d_frame.Name,
    
    mode='markers',
    marker=dict(
        size=12,
        #color= z,          #set color to an array/list of desired value (plotly.ly)
 
        colorscale='Viridis',   #Choose a colorscale
        opacity=0.8
    )
)

data = [trace]
layout = go.Layout(
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0
    )
)
fig = go.Figure(data=data, layout=layout,)

iplot(fig)
```


<div>
        
        
            <div id="ea2d3c24-6f34-4b69-b27f-55a9eaf7def9" class="plotly-graph-div" style="height:525px; width:100%;"></div>
            <script type="text/javascript">
                require(["plotly"], function(Plotly) {
                    window.PLOTLYENV=window.PLOTLYENV || {};
                    window.PLOTLYENV.BASE_URL='https://plot.ly';
                    
                if (document.getElementById("ea2d3c24-6f34-4b69-b27f-55a9eaf7def9")) {
                    Plotly.newPlot(
                        'ea2d3c24-6f34-4b69-b27f-55a9eaf7def9',
                        [{"marker": {"colorscale": "Viridis", "opacity": 0.8, "size": 12}, "mode": "markers", "text": ["Cristiano Ronaldo", "L. Messi", "Neymar", "L. Su\u00e1rez", "M. Neuer", "R. Lewandowski", "De Gea", "E. Hazard", "T. Kroos", "G. Higua\u00edn"], "type": "scatter3d", "uid": "89dfc31a-e514-4984-b56b-3e30f31f5d8b", "x": ["93", "95", "95", "91", "48", "89", "42", "92", "89", "85"], "y": ["91", "97", "96", "86", "30", "85", "18", "93", "79", "84"], "z": ["94", "85", "80", "87", "25", "88", "31", "79", "87", "88"]}],
                        {"margin": {"b": 0, "l": 0, "r": 0, "t": 0}},
                        {"showLink": false, "linkText": "Export to plot.ly", "plotlyServerURL": "https://plot.ly", "responsive": true}
                    ).then(function(){
                            
var gd = document.getElementById('ea2d3c24-6f34-4b69-b27f-55a9eaf7def9');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };
                });
            </script>
        </div>

In above visualization, we are trying to visualize the top ten players performance features (Ball Control,dribling and shot power ).If we hover over the graph, we can visualize that messi,ronaldo,Hazard and neymar ball control and dribbling are aproximately same but in the case of shot power ronaldo is much better.Furthermore,Neuer and Gea are having worst ball control,dribling and shot power performance features.