const title = 'Graph-Structured Policy Learning for Multi-Goal Manipulation Tasks'
const authors = [
	{'name' : 'David M. Klee', link : 'https://dmklee.github.io'},
   	{'name' : 'Ondrej Biza', link : 'https://sites.google.com/view/obiza'},
	{'name' : 'Robert Platt', link : 'https://www.khoury.northeastern.edu/people/robert-platt/'}
]
const associations = [
	{'name' : 'Khoury College at Northeastern University',
	 'link' : 'https://www.khoury.northeastern.edu/',
	 'logo' : 'assets/khoury_logo.png',
	},
]

function make_header(name) {
	body.append('div')
		.style('margin', '30px 0 10px 0')
		.style('padding-left', '8px')
		.style('padding-bottom', '4px')
		.style('border-bottom', '1px #555 solid')
		.style('width', '100%')
		.append('p')
		.style('font-size', '1.5rem')
		.style('font-style', 'italic')
		.style('margin', '2px 4px')
		.text(name)
}

const max_width = '800px';

var body = d3.select('body')
			 .style('max-width', max_width)
			 .style('margin', '60px auto')
			 .style('margin-top', '100px')
			 .style("font-family", "Garamond")
			 .style("font-size", "1.2rem")

// title
body.append('p')
	.style('font-size', '2.2rem')
	.style('font-weight', 600)
	.style('text-align', 'center')
	.style('margin', '20px auto')
	.text(title)

// authors
var authors_div = body.append('div').attr('class', 'flex-row')
for (let i=0; i < authors.length; i++) {
	authors_div.append('a')
				.attr('href', authors[i]['link'])
				.text(authors[i]['name'])
				.style('margin', '10px')
}

// associations
var associations_div = body.append('div').attr('class', 'flex-row')
for (let i=0; i < associations.length; i++) {
	associations_div.append('a')
					.attr('href', associations[i]['link'])
					.append('img')
					.attr('src', associations[i]['logo'])
					.style('height', '70px')
}

// abstract
body.append('div')
	.style('width', '80%')
	.style('margin', '10px auto')
	.style('text-align', 'justify')
	.append('span').style('font-weight', 'bold').text('Abstract: ')
	.append('span').style('font-weight', 'normal')
	.text('Multi-goal policy learning for robotic manipulation is challenging.  Prior successes have used state-based representations of the objects or provided demonstration data to facilitate learning. In this paper, by hand-coding a high-level discrete representation of the domain, we show that policies to reach dozens of goals can be learned with a single network using Q-learning from pixels.  The agent focuses learning on simpler, local policies which are sequenced together by planning in the abstract space.  We compare our method against standard multi-goal RL baselines, as well as other methods that leverage the discrete representation, on a challenging block construction domain. We find that our method can build more than a hundred different block structures, and demonstrate forward transfer to structures with novel objects.  Lastly, we deploy the policy learned in simulation on a real robot.')

make_header('Paper')
body.append('div').style('font-weight', 'bold').text(title)
	.append('div').style('font-weight', 'normal').text(authors.map(d => ' '+d.name))
	.append('div').style('font-style', 'italic').text("IROS'22")
	.append('div').style('font-style', 'normal').append('a').attr('href', 'https://arxiv.org/abs/2207.11313').text('[Arxiv]')
	

make_header('Video Summary')
body.append('iframe')
	.attr('width', '620px')
	.attr('height', '400px')
	.attr('src', './assets/video.mp4')

make_header('Code')
body.append('div').attr('class', 'content')
	.text('View the code on Github ')
	.append('a')
	.attr('href', 'https://github.com/dmklee/graph-structured-manip')
	.text('here.')

make_header('Citation')
body.append('div').attr('class', 'content')
	.append('p')
	.style('border-radius', '6px')
	.style('padding', '10px')
	.style('background-color', '#eee')
	.append('pre')
	.style('font-size', '0.8rem')
	.text(`@misc{graphstructured2022,
  title = {Graph-Structured Policy Learning for Multi-Goal Manipulation Tasks},
  author = {Klee, David and Biza, Ondrej and Platt, Robert},
  url = {https://arxiv.org/abs/2207.11313},
  publisher = {arXiv},
  year = {2022},
}`)

// common syntax
body.selectAll('.flex-row')
	.style('margin', '20px auto')
    .style('display', 'flex')
    .style('justify-content', 'center')
    .style('flex-direction', 'row')
    .style('width', '100%')
body.selectAll('a').style('color', 'blue')
body.selectAll('.content')
	.style('margin', '20px auto')
	.style('width', '90%')
