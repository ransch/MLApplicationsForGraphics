const path = require('path')
const express = require('express')
const cors = require('cors')
const app = express()

app.set('port', process.env.PORT || 80);
app.set('views', __dirname + '/views');
app.set('view engine', 'ejs');
app.engine('html', require('ejs').renderFile);
app.use(cors())
app.use(express.static(path.join(__dirname, 'public')));

app.get('/', (req, res) => {
  res.render('index.html')
})

app.listen(app.get('port'), () => {
    console.log(`Express server listening on port ${app.get('port')}`)
})
