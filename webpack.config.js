'use strict';

var webpack = require('webpack');

module.exports = {
  module: {
    loaders: [
      { test: /\.glsl$/, loader: 'raw' },
      { test: /\.json$/, loader: 'json' },
      { test: /\.js$/, loaders: ['babel-loader'], exclude: /node_modules/ }
    ]
  },
  plugins: [
    new webpack.optimize.OccurenceOrderPlugin(),
    new webpack.optimize.UglifyJsPlugin({
      compressor: {
        screw_ie8: true,
        warnings: false
      }
    })
  ]
};
