d3.json("data/network.json", function(error, graph) {
	if (error) throw error;

	var margin = {top: 70, right: 30, bottom: 50, left: 80},
		width = 0.5 * window.innerWidth - margin.left - margin.right,
		height = 0.5 * width - margin.top - margin.bottom;

	var svg = d3.select("#network_plot").append("svg")
		.attr("width", width + margin.left + margin.right)
		.attr("height", height + margin.top + margin.bottom)
		.append("g")
		.attr("transform", "translate(" + margin.left + "," + margin.top + ")");

	var div = d3.select("#network_plot").append("div")
		.attr("class", "tooltip")
		.style("opacity", 0)
		.text("-");

	//var color = d3.scaleOrdinal(d3.schemeCategory20);

	var simulation = d3.forceSimulation()
		.force("link", d3.forceLink().id(function(d,i) { return i; }))
		.force("charge", d3.forceManyBody())
		.force("center", d3.forceCenter(width / 2, height / 2));

	var link = svg.append("g")
		.attr("class", "links")
		.selectAll("line")
		.data(graph.links)
		.enter().append("line");
		//.attr("stroke-width", function(d) { return Math.sqrt(d.value); });

	var node = svg.append("g")
		.attr("class", "nodes")
		.selectAll("circle")
		.data(graph.nodes)
		.enter().append("circle")
		.attr("r", function(d) { return d.rad; })
		.attr("fill", function(d) { return d.color; })
		.call(d3.drag()
		.on("start", dragstarted)
		.on("drag", dragged)
		.on("end", dragended));

	d3.selectAll('.nodes')
		.data(graph.nodes)

	/*node.append("title")
		.text(function(d) { return d.id; });*/

	svg.append("text")
		.attr("class", "ctitle")
		.attr("x", (width / 2))
		.attr("y", -margin.top/2)
		.attr("fill","white")
		.attr("text-anchor", "middle")
		.text("Interactive Subgraph with Predicted PPI");

	node.on("dblclick", dblclick)
		.on("mouseover", function(d) {
			div.transition()
				.duration(150)
				.style("opacity", .9);
			console.log(d.symb)
			div.html("<strong>Entrez ID:</strong>&nbsp;" + d.id + "&emsp; <strong>Symbol:</strong>&nbsp;" + d.symb + "&emsp; <strong>Name:</strong>&nbsp;" + d.protname)
			/*d3.select(this).transition()
				.duration(150)
				.attr("r", 7);*/
		})
		.on("mouseout", function(d) {
			div.transition()
				.duration(500)
				.style("opacity", 0);
			/*d3.select(this).transition()
				.duration(500)
				.attr("r", 3);*/
		});

	simulation.nodes(graph.nodes)
		.on("tick", ticked);

	simulation.force("link")
		.links(graph.links);

	function dblclick(d){
		window.open("https://www.ncbi.nlm.nih.gov/gene/"+d.id, '_blank');
	}

	function ticked() {
		link.attr("x1", function(d) { return d.source.x; })
			.attr("y1", function(d) { return d.source.y; })
			.attr("x2", function(d) { return d.target.x; })
			.attr("y2", function(d) { return d.target.y; });

		node.attr("cx", function(d) { return d.x; })
			.attr("cy", function(d) { return d.y; });
	}


	function dragstarted(d) {
		if (!d3.event.active) simulation.alphaTarget(0.3).restart();
		d.fx = d.x;
		d.fy = d.y;
	}

	function dragged(d) {
		d.fx = d3.event.x;
		d.fy = d3.event.y;
	}

	function dragended(d) {
		if (!d3.event.active) simulation.alphaTarget(0);
		d.fx = null;
		d.fy = null;
	}

});
