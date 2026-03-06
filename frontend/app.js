async function predict(){

const amount = document.getElementById("amount").value;

const response = await fetch("http://127.0.0.1:8000/predict", {

method:"POST",

headers:{
"Content-Type":"application/json"
},

body:JSON.stringify({
features:[amount]
})

});

const data = await response.json();

document.getElementById("result").innerText =
"Prediction: " + data.prediction;

}

/* Chart */

const ctx = document.getElementById("fraudChart");

new Chart(ctx, {

type:"pie",

data:{
labels:["Normal","Fraud"],

datasets:[{
data:[284315,492],

backgroundColor:[
"#22c55e",
"#ef4444"
]

}]
}

});
