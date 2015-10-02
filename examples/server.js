'use strict';

let express = require('express');
let app = express();

// set proper headers for json and gzipped-json files
app.use(express.static(__dirname, {
  setHeaders: function(res, path, stat) {
    if (path.endsWith('.json.gz')) {
      res.set({
        'Content-Encoding': 'gzip',
        'Content-Type': 'application/json'
      });
    } else if (path.endsWith('.json')) {
      res.set('Content-Type', 'application/json');
    }
  }
}));

app.listen(8000, function() {
  console.log('Examples server running at http://localhost:8000');
});
