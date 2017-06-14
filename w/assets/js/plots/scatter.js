function scatterplotexample() {

	//Dynamic, random dataset
	var dataset = [];
	var numDataPoints = 50;
	var xRange = Math.random() * 1000;
	var yRange = Math.random() * 1000;
	for (var i = 0; i < numDataPoints; i++) {
		var newNumber1 = Math.floor(Math.random() * xRange);
		var newNumber2 = Math.floor(Math.random() * yRange);
		dataset.push([newNumber1, newNumber2]);
	}

	//Width and height
	var w = 1000;
	var h = 300;
	var padding = 50;

	var xScale = d3.scale.linear()
		.domain([0, d3.max(dataset, function(d) { return d[0]; })])
		.range([padding, w - padding*2])
		.nice();

	var yScale = d3.scale.linear()
		.domain([0, d3.max(dataset, function(d) { return d[1]; })])
		.range([h - padding, padding]);

	var rScale = d3.scale.linear()
		.domain([0, d3.max(dataset, function(d) { return d[1]; })])
		.range([2, 10]);

	var svg = d3.select("#results")
		.append("svg")
		.attr("width", w)
		.attr("height", h);

	var xAxis = d3.svg.axis()
		.scale(xScale)
		.orient("bottom")
		.ticks(5);

	var yAxis = d3.svg.axis()
		.scale(yScale)
		.orient("left")
		.ticks(5);

	//Define clipping path
	svg.append("clipPath")                  //Make a new clipPath
		.attr("id", "chart-area")           //Assign an ID
		.append("rect")                     //Within the clipPath, create a new rect
		.attr("x", padding)                 //Set rect's position and size…
		.attr("y", padding)
		.attr("width", w - padding * 3)
		.attr("height", h - padding * 2);

	//Create circles
	svg.append("g")                             //Create new g
		.attr("id", "circles")                   //Assign ID of 'circles'
		.attr("clip-path", "url(#chart-area)")   //Add reference to clipPath
		.selectAll("circle")                     //Continue as before…
		.data(dataset)
		.enter()
		.append("circle")
		.append("circle")
		.attr("cx", function(d) {
			return xScale(d[0]);
		})
		.attr("cy", function(d) {
			return yScale(d[1]);
		})
		.attr("r", 3)
		.attr("fill", "white");

	svg.append("g")
		.attr("class", "x axis")
		.attr("transform", "translate(0," + (h - padding) + ")")
		.call(xAxis);

	svg.append("g")
		.attr("class", "y axis")
		.attr("transform", "translate(" + padding + ",0)")
		.call(yAxis);


	//On click, update with new data
	d3.select("#results .button")
		.on("click", function() {
			//New values for dataset
			var numValues = dataset.length;						 		//Count original length of dataset
			var maxRange = Math.random() * 1000;						//Max range of new values
			dataset = [];  						 				 		//Initialize empty array
			for (var i = 0; i < numValues; i++) {				 		//Loop numValues times
				var newNumber1 = Math.floor(Math.random() * maxRange);	//New random integer
				var newNumber2 = Math.floor(Math.random() * maxRange);	//New random integer
				dataset.push([newNumber1, newNumber2]);					//Add new number to array
			}

			//Update scale domains
			xScale.domain([0, d3.max(dataset, function(d) { return d[0]; })]);
			yScale.domain([0, d3.max(dataset, function(d) { return d[1]; })]);

			//Update all circles
			svg.selectAll("circle")
				.data(dataset)
				.transition()    // <-- Transition #1
				.duration(1000)
				.each("start", function() {
					d3.select(this)
						.attr("fill", "magenta")
						.attr("r", 7);
				})
				.attr("cx", function(d) {
					return xScale(d[0]);
				})
				.attr("cy", function(d) {
					return yScale(d[1]);
				})
				.transition()    // <-- Transition #2
				.duration(1000)
				.attr("fill", "white")
				.attr("r", 2);
			svg.select(".x.axis")
				.transition()
				.duration(1000)
				.call(xAxis);
			svg.select(".y.axis")
				.transition()
				.duration(1000)
				.call(yAxis);
		});
};

scatterplotexample()
