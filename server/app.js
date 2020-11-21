const express = require('express')


const path = require("path");
const app = express()


const { PythonShell } = require('python-shell')

const bodyParser = require('body-parser');
const urlencodedParser = bodyParser.urlencoded({ extended: true, limit:"50000mb"})

app.use(express.json({limit: '50000mb'}));

const publicPath = path.join(__dirname, "../public");
const viewPath = path.join(publicPath + "/views");

app.use(express.static(publicPath));
app.set("views", viewPath);
app.set("view engine", "ejs");
app.get('/', (req, res) => {
    res.render('index')
})

// app.get('/lul', (req, res) => {
//     PythonShell.run('./server/pyfol/hello.py', null, function (err, result) {
//         if (err) throw err;
//         // result is an array consisting of messages collected  
//         // during execution of script.
//         console.log('result: ', result.toString());
//         res.send(`${result.toString()}`);
//     })
// })
// app.get('/testpizza', (req, res) => {
//     let options = {args: ['test']}
//     PythonShell.run('./server/pyfol/pizza_notpizza.py', options , function (err, result) {
//         if (err) throw err;
//         let str = `<img src='https://media-cdn.tripadvisor.com/media/photo-s/0e/54/ea/06/pizza-hut.jpg'><br><h1>${result.toString()}</h1>`
//         res.send(`${str}`);
//     })
// })

// app.get('/testoval', (req, res) => {

//     PythonShell.run('./server/pyfol/oval_check.py', null, function (err, result) {
//         if (err) throw err;
//         let str = `<img src="/static/img/capsule1.jpg"><br><h1>${result.toString()}</h1>`
//         res.send(`${str}`);
//     })
// })


app.get('/pizza', (req, res) => {
    res.render('pizza')
})
app.post('/predict_pizza', urlencodedParser, (req, res,) => {
    let options = {"args": [req.body.img]}
    PythonShell.run('./server/pyfol/pizza_notpizza.py', options, function (err, result) {
        if (err) throw err;
        let str = `${result.toString()}`
        console.log(str)
        res.send(`${str}`);
    })
    // res.send(`yooyoy`)
})

app.get('/drug', (req,res)=>{
    res.render('drug')
})
app.post('/predict_drug', urlencodedParser, (req, res,) => {
    let options = {"args": [req.body.img]}
    PythonShell.run('server/pyfol/shape_predict/func/fed.py', options, function (err, result) {
        if (err) throw err;
        let str = `${result.toString()}`
        // console.log(str)
        res.send(`${str}`);
    })
    // res.send(`yooyoy`)
})

app.use('/static', express.static(publicPath))

app.listen(3000, function () {

})
