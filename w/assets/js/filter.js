d3.csv("logwebsite.csv", function(error, data) {

	var margin = {top: 20, right: 20, bottom: 30, left: 40},
		width = 960 - margin.left - margin.right,
		height = 500 - margin.top - margin.bottom;

	var x = d3.scale.linear()
		.range([0, width]);

	var y = d3.scale.linear()
		.range([height, 0]);


	var svg = d3.select("#discussion").append("svg")
		.attr("width", width + margin.left + margin.right)
		.attr("height", height + margin.top + margin.bottom)
		.append("g")
		.attr("transform", "translate(" + margin.left + "," + margin.top + ")");


	// Coerce the strings to numbers.
	data.forEach(function(d) {
		d.epoch = +d.epoch;
		d.max_epochs = +d.max_epochs;
		d.mcc = +d.mcc;
		});

	// Compute the scales’ domains.
	x.domain(d3.extent(data, function(d) { return d.epoch; })).nice();
	y.domain(d3.extent(data, function(d) { return d.mcc; })).nice();

	// Add the x-axis.
	svg.append("g")
		.attr("class", "x axis")
		.attr("transform", "translate(0," + height + ")")
		.call(d3.svg.axis().scale(x).orient("bottom"));

	// Add the y-axis.
	svg.append("g")
		.attr("class", "y axis")
		.call(d3.svg.axis().scale(y).orient("left"));

	// Get a subset of the data based on the group
	function getFilteredData(data, optim, input_mode, max_epochs) {
		return data.filter(function(point) { return point.max_epochs === parseInt(max_epochs); })
					.filter(function(point) { return point.input_mode === input_mode; })
					.filter(function(point) { return point.optim === optim; });
	}

	// Helper function to add new points to our data
	function enterPoints(data) {
		// Add the points!
		svg.selectAll(".point")
			.data(data)
			.enter().append("path")
			.attr("class", "point")
			.attr('fill', 'white')
			.attr("d", d3.svg.symbol().type("circle"))
			.attr("transform", function(d) { return "translate(" + x(d.epoch) + "," + y(d.mcc) + ")"; });
	}

	function exitPoints(data) {
		svg.selectAll(".point")
			.data(data)
			.exit()
			.remove();
	}

	function updatePoints(data) {
		svg.selectAll(".point")
			.data(data)
			.transition()
			.ease('elastic')
			.duration(1000)
			.attr("transform", function(d) { return "translate(" + x(d.epoch) + "," + y(d.mcc) + ")"; });
	}

	// New select element for allowing the user to select a group!
	var $max_epochsSelector = document.querySelector('#maxepochs-select');
	var inmodbutt = d3.select('input[name="input-mode"]:checked').property("value");
	var optimbutt = d3.select('input[name="optim"]:checked').property("value");
	var groupData = getFilteredData(data, optimbutt, inmodbutt, $max_epochsSelector.value);

	// Enter initial points filtered by default select value set in HTML
	enterPoints(groupData);

	$max_epochsSelector.onchange = function(e) {
		var max_epochs = e.target.value;
		var input_mode = d3.select('input[name="input-mode"]:checked').property("value");
		optim = d3.select('input[name="optim"]:checked').property("value");
		var groupData = getFilteredData(data, optim, input_mode, max_epochs);

		updatePoints(groupData);
		enterPoints(groupData);
		exitPoints(groupData);

	};

	d3.selectAll("input")
		.on("change", function(e) {
		var max_epochs = $max_epochsSelector.value;
		var input_mode = d3.select('input[name="input-mode"]:checked').property("value");
		optim = d3.select('input[name="optim"]:checked').property("value");
		var groupData = getFilteredData(data, optim, input_mode, max_epochs);

		updatePoints(groupData);
		enterPoints(groupData);
		exitPoints(groupData);

	});
});
