const HtmlWebpackPlugin = require("html-webpack-plugin");
const path = require("path");

module.exports = {
  entry: "./index.jsx",
  output: {
    path: path.resolve(__dirname, "dist"),
    filename: "bundle.[hash].js",
  },
  devServer: {
    compress: true,
    port: 8080,
    hot: true,
    static: "./dist",
    historyApiFallback: true,
    open: true,
  },
  module: {
    rules: [
      {
        test: /\.(js|jsx)$/,
        exclude: /node_modules/,
        use: {
          loader: "babel-loader",
        },
      },
    ],
  },
  experiments: {
    asyncWebAssembly: true,
  },
  plugins: [
    new HtmlWebpackPlugin({
      template: __dirname + "/index.html",
      filename: "index.html",
    }),
  ],
  mode: "development",
  devtool: "inline-source-map",
};