export default (req, res) => {
  console.log("HELLO HIT");
  res.status(200).json({ ok: true, route: "/api/hello" });
};
