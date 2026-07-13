module.exports = (req, res) => {
  console.log("TEST HIT");
  res.status(200).json({ ok: true, route: "/api/test" });
};
