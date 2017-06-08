var sample_data = [
	{"value": 100, "weight": .45, "type": "alpha"},
	{"value": 70, "weight": .60, "type": "beta"},
	{"value": 40, "weight": -.2, "type": "gamma"},
	{"value": 15, "weight": .1, "type": "delta"}
]
var visualization = d3plus.viz()
	.container("#viz")
	.data(sample_data)
	.type("scatter")
	.id("type")
	.x("value")
	.y("weight")
	.size("value")
	.shape("donut")   // set shape of data display
	.draw()
