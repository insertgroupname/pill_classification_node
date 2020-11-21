const express = require('express')


const path = require("path");
const app = express()
const crypto = require("crypto")

const { PythonShell } = require('python-shell')

const bodyParser = require('body-parser');
const urlencodedParser = bodyParser.urlencoded({ extended: true, limit: "50000mb" })

app.use(express.json({ limit: '50000mb' }));

const publicPath = path.join(__dirname, "../public");
const viewPath = path.join(publicPath + "/views");

app.use(express.static(publicPath));
app.set("views", viewPath);
app.set("view engine", "ejs");
app.get('/', (req, res) => {
    res.render('index')
})

app.get('/pizza', (req, res) => {
    res.render('pizza')
})
app.post('/predict_pizza', urlencodedParser, (req, res,) => {
    let options = { "args": [req.body.img] }
    PythonShell.run('./server/pyfol/pizza_notpizza.py', options, function (err, result) {
        if (err) throw err;
        let str = `${result.toString()}`
        console.log(str)
        res.send(`${str}`);
    })
})



const multer = require('multer');

var storage = multer.diskStorage({
    destination: function (req, file, cb) {
        cb(null, publicPath+'/img/uploads')
    },
    filename: function (req, file, cb) {
        var current_date = (new Date()).valueOf().toString();
        var random = Math.random().toString();
        var hashed = crypto.createHash('sha1').update(current_date + random).digest('hex')
        console.log(hashed)
        cb(null, hashed+"-"+(file.mimetype).replace('/','.'))
    }
})

var upload = multer({ storage: storage })

app.post('/upload', upload.single('imgInput'), (req, res, next) => {
    const file = req.file
    if (!file) {
        const error = new Error('Please upload a file')
        error.httpStatusCode = 400
        return next(error)
    }
    console.log(req.file.filename);
    res.send(file)
})

app.get('/drug', (req, res) => {
    res.render('drug')
})
app.post('/predict_drug', upload.single('imgInput'), urlencodedParser, (req, res,) => {
    const file = req.file
    let options = { "args": [req.file.filename] }
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
