d3.csv("logwebsite.csv", function(error, data) {

	function makesvg(plotname) {
		var plot = d3.select("#discussion").append("svg")
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
			.enter().append("path")
			.attr("class", "point")
			.attr('fill', 'white')
			.attr("d", d3.svg.symbol().type("circle"))
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

	function getFilteredData(data, optim, input_mode, class_weights) {
		return data.filter(function(point) { return point.class_weight === class_weights; })
					.filter(function(point) { return point.input_mode === input_mode; })
					.filter(function(point) { return point.optim === optim; });
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
		height = width - margin.top - margin.bottom;

	var x = d3.scale.linear()
		.range([0, width]);

	var y = d3.scale.linear()
		.range([height, 0]);

	var xAxis = d3.svg.axis()
		.scale(x)
		.orient("bottom")
		.tickFormat(d3.format("d"))
		.ticks(5);

	var yAxis = d3.svg.axis()
		.scale(y)
		.orient("left");

	var class_weights = d3.select("#classweights-select").node().value;
	var inmodbutt = d3.select('input[name="input-mode"]:checked').property("value");
	var optimbutt = d3.select('input[name="optim"]:checked').property("value");

	var groupData = getFilteredData(data, optimbutt, inmodbutt, class_weights);

	var mccplot = makesvg("mccplot")
	var f1plot = makesvg("f1plot")
	d3.select("#discussion").append("br")
	d3.select("#discussion").append("br")
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
		var class_weights = d3.select("#classweights-select").node().value;
		var input_mode = d3.select('input[name="input-mode"]:checked').property("value");
		var optim = d3.select('input[name="optim"]:checked').property("value");
		var groupData = getFilteredData(data, optim, input_mode, class_weights);

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
