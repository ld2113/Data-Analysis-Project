d3.csv("logwebsite.csv", function(error, data) {

	function makesvg(plotname) {
		var plot = d3.select(".nnplots").append("svg")
			.attr("class", "nnsvg")
			.attr("id", plotname)
			.attr("width", width + margin.left + margin.right)
			.attr("height", height + margin.top + margin.bottom)
			.append("g")
			.attr("transform", "translate(" + margin.left + "," + margin.top + ")");

		return plot
	}

	function createAxis(data, plot, txt, yvar) {
		y.domain([0, d3.max(data, function(d) { return eval("d."+yvar); })]).nice();

		plot.append("g")
			.attr("class", "y axis")
			.call(yAxis);

		plot.append("g")
			.attr("class", "x axis")
			.attr("transform", "translate(0," + height + ")")
			.call(xAxis);

		plot.append("text")
			.attr("class", "ctitle")
			.attr("x", (width / 2))
			.attr("y", -margin.top/2)
			.attr("fill","white")
			.attr("text-anchor", "middle")
			.text(txt);

		plot.append("text")
			.attr("class", "x label")
			.attr("x", (width / 2))
			.attr("y", height + margin.bottom/1.25)
			.attr("fill","white")
			.attr("text-anchor", "middle")
			.attr("font-size", "0.8em")
			.text("Epoch Number");

		plot.append("text")
			.attr("class", "y label")
			.attr("x", -height/2)
			.attr("y", -margin.left/1.5)
			.attr("fill","white")
			.attr("text-anchor", "middle")
			.attr("font-size", "0.8em")
			.attr("transform", "rotate(-90)")
			.text(txt);
	}

	function enterPoints(data, groupData, plot, yvar) {
		x.domain([0, d3.max(groupData, function(d) { return d.epoch; })]).nice();
		y.domain([0, d3.max(data, function(d) { return eval("d."+yvar); })]).nice();

		plot.selectAll(".point")
			.data(groupData)
			.enter().append("circle")
			.attr("class", "point")
			.attr('fill', 'white')
			.attr("r", 5)
			.attr("transform", function(d) { return "translate(0," + height + ")"; })
			.transition()
			.duration(1000)
			.attr("transform", function(d) { return "translate(" + x(d.epoch) + "," + y(eval("d."+yvar)) + ")"; });
	}

	function exitPoints(data, plot) {
		plot.selectAll(".point")
			.data(data)
			.exit()
			.transition()
			.duration(1000)
			.attr("transform", function(d) { return "translate(0," + height + ")"; })
			.remove();
	}

	function updatePoints(data, groupData, plot, yvar) {
		x.domain([0, d3.max(groupData, function(d) { return d.epoch; })]).nice();
		y.domain([0, d3.max(data, function(d) { return eval("d."+yvar); })]).nice();

		plot.selectAll(".point")
			.data(groupData)
			.transition()
			.duration(1000)
			.attr("transform", function(d) { return "translate(" + x(d.epoch) + "," + y(eval("d."+yvar)) + ")"; });

		plot.select(".x.axis")
			.transition()
			.duration(1000)
			.call(xAxis);
	}

	function getFilteredData(data, inputs, structure, autodims, regulbutt, drop) {
		return data.filter(function(d) { return d.input_mode === inputs; })
					.filter(function(d) { return d.concat_struct === structure; })
					.filter(function(d) { return d.dom_path === autodims; })
					.filter(function(d) { return d.drop === drop; })
					.filter(function(d) { return d.reg === regulbutt; });
	}


////////////////////////////////// MAIN PROGRAM ////////////////////////////////

	document.getElementById("nnform").reset();

	data.forEach(function(d) {
		d.epoch = +d.epoch;
		d.mcc = +d.mcc;
		d.f1 = +d.f1;
		d.prec = +d.prec;
		d.recall = +d.recall;
		});

	var margin = {top: 30, right: 50, bottom: 50, left: 70},
		width = 0.25 * window.innerWidth - margin.left - margin.right,
		height = 0.8*width - margin.top - margin.bottom;

	var x = d3.scaleLinear()
		.range([0, width]);

	var y = d3.scaleLinear()
		.range([height, 0]);

	var xAxis = d3.axisBottom(x)
		.tickFormat(d3.format("d"))
		.ticks(5);

	var yAxis = d3.axisLeft(y)
		.ticks();

	var inputs = d3.select("#input-select").node().value;
	var structure = d3.select("#structure-select").node().value;
	var autodims = d3.select('input[name="dimens"]:checked').property("value");
	var regulbutt = d3.select('input[name="regul"]:checked').property("value");
	var drop = d3.select('input[name="drop"]:checked').property("value");

	var groupData = getFilteredData(data, inputs, structure, autodims, regulbutt, drop);

	var mccplot = makesvg("mccplot")
	var f1plot = makesvg("f1plot")
	var precplot = makesvg("precplot")
	var recplot = makesvg("recplot")

	x.domain([0, d3.max(groupData, function(d) { return d.epoch; })]).nice();

	createAxis(data, mccplot, "Matthews Correlation Coefficient", "mcc");
	createAxis(data, f1plot, "F1 Score", "f1");
	createAxis(data, recplot, "Recall", "recall");
	createAxis(data, precplot, "Precision", "prec");

	enterPoints(data, groupData,mccplot, "mcc");
	enterPoints(data, groupData,f1plot, "f1");
	enterPoints(data, groupData,precplot, "prec");
	enterPoints(data, groupData,recplot, "recall");


	d3.selectAll("input,select")
		.on("change", function(e) {
		var inputs = d3.select("#input-select").node().value;
		var structure = d3.select("#structure-select").node().value;
		var autodims = d3.select('input[name="dimens"]:checked').property("value");
		var regulbutt = d3.select('input[name="regul"]:checked').property("value");
		var drop = d3.select('input[name="drop"]:checked').property("value");

		var groupData = getFilteredData(data, inputs, structure, autodims, regulbutt, drop);

		updatePoints(data, groupData, mccplot, "mcc");
		enterPoints(data, groupData, mccplot, "mcc");
		exitPoints(groupData, mccplot);

		updatePoints(data, groupData, f1plot, "f1");
		enterPoints(data, groupData, f1plot, "f1");
		exitPoints(groupData, f1plot);

		updatePoints(data, groupData, precplot, "prec");
		enterPoints(data, groupData, precplot, "prec");
		exitPoints(groupData, precplot);

		updatePoints(data, groupData, recplot, "recall");
		enterPoints(data, groupData, recplot, "recall");
		exitPoints(groupData, recplot);

	});
	/*
	DOESNT WORK!!
	d3.select("window").on("window.resize", function() {

		margin = {top: 20, right: 20, bottom: 30, left: 40},
			width = 0.5 * window.innerWidth - margin.left - margin.right,
			height = 0.5*width - margin.top - margin.bottom;

		x = d3.scale.linear()
			.range([0, width]);

		y = d3.scale.linear()
			.range([height, 0]);

		updatePoints(groupData);
		enterPoints(groupData);
		exitPoints(groupData);
	});
	*/
});
